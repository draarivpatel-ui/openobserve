// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{
            ArrayBuilder, ArrayRef, GenericStringBuilder, LargeStringArray, OffsetSizeTrait,
            StringArray, StringViewArray, StringViewBuilder,
        },
        datatypes::{DataType, Field},
    },
    common::{
        Result, ScalarValue, exec_datafusion_err,
        types::{NativeType, logical_int64, logical_null, logical_string},
    },
    logical_expr::{
        Coercion, ColumnarValue, ScalarFunctionArgs, ScalarFunctionImplementation, ScalarUDF,
        ScalarUDFImpl, Signature, TypeSignatureClass, Volatility, ptr_eq::PtrEq,
        type_coercion::binary::string_coercion,
    },
};

/// Rebinds each `ArrayRef` ident to the concrete array of `$data_type`, then evaluates `$body`
/// once per string type, so `$body` can call a function generic over `StringArrayType`.
/// With `$builder =>`, `$builder` names the `StringValuesBuilder` of that same type.
macro_rules! with_string_arrays {
    ($data_type:expr, [$($array:ident),+ $(,)?], $body:expr) => {{
        use ::datafusion::arrow::{array::AsArray as _, datatypes::DataType};
        match $data_type {
            DataType::Utf8 => {
                $(let $array = $array.as_string::<i32>();)+
                $body
            }
            DataType::LargeUtf8 => {
                $(let $array = $array.as_string::<i64>();)+
                $body
            }
            DataType::Utf8View => {
                $(let $array = $array.as_string_view();)+
                $body
            }
            other => ::datafusion::common::exec_err!("expected a string array, got {other}"),
        }
    }};
    ($data_type:expr, [$($array:ident),+ $(,)?], $builder:ident => $body:expr) => {{
        use ::datafusion::arrow::{
            array::{AsArray as _, GenericStringBuilder, StringViewBuilder},
            datatypes::DataType,
        };
        match $data_type {
            DataType::Utf8 => {
                type $builder = GenericStringBuilder<i32>;
                $(let $array = $array.as_string::<i32>();)+
                $body
            }
            DataType::LargeUtf8 => {
                type $builder = GenericStringBuilder<i64>;
                $(let $array = $array.as_string::<i64>();)+
                $body
            }
            DataType::Utf8View => {
                type $builder = StringViewBuilder;
                $(let $array = $array.as_string_view();)+
                $body
            }
            other => ::datafusion::common::exec_err!("expected a string array, got {other}"),
        }
    }};
}

pub(crate) use with_string_arrays;

/// How a UDF's return type follows the common type of its string arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReturnType {
    Fixed(DataType),
    /// The common string type, as DataFusion's `lower` and `split_part` return.
    String,
    /// A list of the common string type, as DataFusion's `regexp_match` returns.
    StringList,
}

impl ReturnType {
    fn resolve(&self, string_type: DataType) -> DataType {
        match self {
            Self::Fixed(data_type) => data_type.clone(),
            Self::String => string_type,
            Self::StringList => string_list_type(string_type),
        }
    }
}

/// A builder of string values for list and struct outputs of any string type.
pub trait StringValuesBuilder: ArrayBuilder + Default {
    fn append_str(&mut self, value: &str);
    fn append_str_option(&mut self, value: Option<&str>);
}

impl<O: OffsetSizeTrait> StringValuesBuilder for GenericStringBuilder<O> {
    fn append_str(&mut self, value: &str) {
        self.append_value(value);
    }

    fn append_str_option(&mut self, value: Option<&str>) {
        self.append_option(value);
    }
}

impl StringValuesBuilder for StringViewBuilder {
    fn append_str(&mut self, value: &str) {
        self.append_value(value);
    }

    fn append_str_option(&mut self, value: Option<&str>) {
        self.append_option(value);
    }
}

#[derive(PartialEq, Eq, Hash)]
struct CoercibleUdf {
    name: String,
    signature: Signature,
    string_args: Vec<usize>,
    return_type: ReturnType,
    fun: PtrEq<ScalarFunctionImplementation>,
}

impl std::fmt::Debug for CoercibleUdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoercibleUdf")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .field("string_args", &self.string_args)
            .field("return_type", &self.return_type)
            .finish_non_exhaustive()
    }
}

impl ScalarUDFImpl for CoercibleUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let string_type = common_string_type(self.string_args.iter().map(|&i| &arg_types[i]))?;
        Ok(self.return_type.resolve(string_type))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        (self.fun)(&args.args)
    }
}

/// Keeps any string type as it is and casts every other type, like `Exact(Utf8)` did.
pub fn string_param() -> Coercion {
    Coercion::new_implicit(
        TypeSignatureClass::Native(logical_string()),
        vec![TypeSignatureClass::Any],
        NativeType::String,
    )
}

pub fn int64_param() -> Coercion {
    Coercion::new_implicit(
        TypeSignatureClass::Native(logical_int64()),
        vec![
            TypeSignatureClass::Integer,
            TypeSignatureClass::Native(logical_null()),
        ],
        NativeType::Int64,
    )
}

pub fn string_signature(arg_count: usize, volatility: Volatility) -> Signature {
    Signature::coercible(vec![string_param(); arg_count], volatility)
}

/// `create_udf` with a coercible signature, so string columns reach `fun` uncast, and a
/// return type that may follow the common type of the `string_param()` arguments.
pub fn create_coercible_udf(
    name: &str,
    params: Vec<Coercion>,
    return_type: ReturnType,
    volatility: Volatility,
    fun: ScalarFunctionImplementation,
) -> ScalarUDF {
    let string_args = (0..params.len())
        .filter(|&i| params[i] == string_param())
        .collect();
    // a coercible signature rejects zero arguments, which `Exact(vec![])` allowed
    let signature = if params.is_empty() {
        Signature::nullary(volatility)
    } else {
        Signature::coercible(params, volatility)
    };
    ScalarUDF::from(CoercibleUdf {
        name: name.to_string(),
        signature,
        string_args,
        return_type,
        fun: fun.into(),
    })
}

/// The type DataFusion's `string_coercion` unifies these string types to; Utf8 when empty.
pub fn common_string_type<'a>(types: impl IntoIterator<Item = &'a DataType>) -> Result<DataType> {
    let mut common: Option<DataType> = None;
    for data_type in types {
        common = Some(match common {
            None => data_type.clone(),
            Some(common) => string_coercion(&common, data_type).ok_or_else(|| {
                exec_datafusion_err!("expected string arguments, got {common} and {data_type}")
            })?,
        });
    }
    Ok(common.unwrap_or(DataType::Utf8))
}

/// Casts the string arguments at `indices` to their common string type, as DataFusion's
/// `replace` does; a scalar costs O(1) to cast. Returns that type and all arguments.
pub fn coerce_string_args(
    args: &[ColumnarValue],
    indices: &[usize],
) -> Result<(DataType, Vec<ColumnarValue>)> {
    let types = indices
        .iter()
        .map(|&i| args[i].data_type())
        .collect::<Vec<_>>();
    let common = common_string_type(&types)?;
    let args = args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            if indices.contains(&i) && arg.data_type() != common {
                arg.cast_to(&common, None)
            } else {
                Ok(arg.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((common, args))
}

/// Collects strings into an array of `string_type`, the type `ReturnType::String` reports.
pub fn collect_strings<T: AsRef<str>>(
    string_type: &DataType,
    values: impl IntoIterator<Item = Option<T>>,
) -> ArrayRef {
    match string_type {
        DataType::Utf8View => Arc::new(values.into_iter().collect::<StringViewArray>()),
        DataType::LargeUtf8 => Arc::new(values.into_iter().collect::<LargeStringArray>()),
        _ => Arc::new(values.into_iter().collect::<StringArray>()),
    }
}

pub fn string_list_type(string_type: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", string_type, true)))
}

/// Value of a string scalar of any string type; `None` for NULL or a non-string scalar.
pub fn scalar_str(value: &ScalarValue) -> Option<&str> {
    match value {
        ScalarValue::Utf8(v) | ScalarValue::LargeUtf8(v) | ScalarValue::Utf8View(v) => v.as_deref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::array::{
            ArrayRef, Int64Array, LargeStringArray, StringArray, StringArrayType, StringViewArray,
        },
        prelude::SessionContext,
    };

    use super::*;

    fn arrays() -> Vec<ArrayRef> {
        let rows = [Some("a"), None, Some("long enough to be out of line")];
        vec![
            Arc::new(StringArray::from_iter(rows)),
            Arc::new(LargeStringArray::from_iter(rows)),
            Arc::new(StringViewArray::from_iter(rows)),
        ]
    }

    fn collect<'a, S: StringArrayType<'a>>(array: S) -> Vec<Option<String>> {
        array.iter().map(|v| v.map(str::to_string)).collect()
    }

    fn dispatch(array: &ArrayRef) -> Result<Vec<Option<String>>> {
        with_string_arrays!(array.data_type(), [array], Ok(collect(array)))
    }

    #[test]
    fn dispatch_reads_every_string_type_in_place() {
        for array in arrays() {
            assert_eq!(
                dispatch(&array).unwrap(),
                vec![
                    Some("a".to_string()),
                    None,
                    Some("long enough to be out of line".to_string())
                ],
                "{}",
                array.data_type()
            );
        }
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(dispatch(&array).is_err());
    }

    #[test]
    fn coerce_string_args_unifies_to_the_widest_string_type() {
        let view = ColumnarValue::Array(arrays()[2].clone());
        let literal = ColumnarValue::Scalar(ScalarValue::Utf8(Some("x".to_string())));
        let int = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));
        let (common, args) =
            coerce_string_args(&[literal.clone(), view.clone(), int.clone()], &[0, 1]).unwrap();
        assert_eq!(common, DataType::Utf8View);
        assert_eq!(args[0].data_type(), DataType::Utf8View);
        assert_eq!(args[1].data_type(), DataType::Utf8View);
        assert_eq!(args[2].data_type(), DataType::Int64);
        let (common, args) = coerce_string_args(&[literal, int], &[0]).unwrap();
        assert_eq!(common, DataType::Utf8);
        assert_eq!(args[0].data_type(), DataType::Utf8);
    }

    /// `data_type` with every Utf8 inside it replaced by `to`.
    fn with_string_type(data_type: &DataType, to: &DataType) -> DataType {
        let field = |f: &Field| {
            f.clone()
                .with_data_type(with_string_type(f.data_type(), to))
        };
        match data_type {
            DataType::Utf8 => to.clone(),
            DataType::List(f) => DataType::List(Arc::new(field(f))),
            DataType::Struct(fields) => {
                DataType::Struct(fields.iter().map(|f| field(f)).collect::<Vec<_>>().into())
            }
            other => other.clone(),
        }
    }

    #[tokio::test]
    async fn builtin_udfs_keep_the_string_type_of_their_input() {
        let rows = [
            Some("[3,1,2]"),
            Some(r#"{"a":"x1","b":{"c":7}}"#),
            Some("Hello world 42, id=17"),
            Some("a,b;c"),
            Some(""),
            None,
        ];
        let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("s", DataType::Utf8, true),
            datafusion::arrow::datatypes::Field::new("v", DataType::Utf8View, true),
            datafusion::arrow::datatypes::Field::new("l", DataType::LargeUtf8, true),
        ]));
        let batch = datafusion::arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter(rows)),
                Arc::new(StringViewArray::from_iter(rows)),
                Arc::new(LargeStringArray::from_iter(rows)),
            ],
        )
        .unwrap();
        let ctx = SessionContext::new();
        crate::datafusion::exec::register_builtin_udfs(&ctx);
        let table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();
        for expr in [
            "str_match({c}, 'world')",
            "str_match_ignore_case({c}, 'HELLO')",
            "match_field({c}, 'x1')",
            "fuzzy_match({c}, 'helo', 1)",
            "re_match({c}, '[0-9]+')",
            "re_not_match({c}, '[0-9]+')",
            "re_matches({c}, '([0-9]+)')",
            "regexp_match_to_fields({c}, '(?<n>[0-9]+)')",
            "spath({c}, 'b.c')",
            "arrcount({c})",
            "arrsort({c})",
            "arr_descending({c})",
            "arrjoin({c}, '-')",
            "arrzip({c}, {c}, ':')",
            "arrindex({c}, 0, 1)",
            "cast_to_arr({c})",
            "string_to_array_v2({c}, ',;')",
            "time_range(1, {c}, {c})",
            "date_format(1700000000000000, {c}, 'UTC')",
            "to_array_string(cast_to_arr({c}))",
            "to_array_string(string_to_array_v2({c}, ','))",
        ] {
            let render = |column: &str| expr.replace("{c}", column);
            let expected = format!("SELECT {} AS r FROM t", render("s"));
            let expected = ctx.sql(&expected).await.unwrap().collect().await.unwrap();
            let expected_type = expected[0].schema().field(0).data_type().clone();
            let expected = datafusion::arrow::util::pretty::pretty_format_batches(&expected)
                .unwrap()
                .to_string();
            for (column, string_type) in [("v", DataType::Utf8View), ("l", DataType::LargeUtf8)] {
                let df = ctx
                    .sql(&format!("SELECT {} AS r FROM t", render(column)))
                    .await
                    .unwrap();
                let plan = df.logical_plan().display_indent().to_string();
                assert!(!plan.contains("CAST"), "{expr} on {column}\n{plan}");
                let actual = df.collect().await.unwrap();
                assert_eq!(
                    actual[0].schema().field(0).data_type(),
                    &with_string_type(&expected_type, &string_type),
                    "{expr} on {column}"
                );
                let actual = datafusion::arrow::util::pretty::pretty_format_batches(&actual)
                    .unwrap()
                    .to_string();
                assert_eq!(actual, expected, "{expr} on {column}");
            }
        }
    }

    #[tokio::test]
    async fn coercible_udf_keeps_string_types_and_casts_others() {
        let udf = create_coercible_udf(
            "arg_type",
            vec![string_param(), int64_param()],
            ReturnType::Fixed(DataType::Utf8),
            Volatility::Immutable,
            Arc::new(|args: &[ColumnarValue]| {
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(format!(
                    "{}/{}",
                    args[0].data_type(),
                    args[1].data_type()
                )))))
            }),
        );
        let ctx = SessionContext::new();
        ctx.register_udf(udf);
        for (arg, expected) in [
            ("arrow_cast('x', 'Utf8View')", "Utf8View/Int64"),
            ("arrow_cast('x', 'LargeUtf8')", "LargeUtf8/Int64"),
            ("'x'", "Utf8/Int64"),
            ("NULL", "Utf8View/Int64"),
            ("1", "Utf8View/Int64"),
        ] {
            let sql = format!("SELECT arg_type({arg}, arrow_cast(1, 'Int32'))");
            let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
            let value = ScalarValue::try_from_array(batches[0].column(0), 0).unwrap();
            assert_eq!(scalar_str(&value), Some(expected), "{sql}");
        }
    }
}
