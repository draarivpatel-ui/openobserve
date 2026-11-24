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

use datafusion::{
    arrow::{
        array::{Array, ArrayIter, AsArray, LargeStringArray, StringArray, StringViewArray},
        datatypes::DataType,
    },
    common::{
        Result, ScalarValue, exec_err,
        types::{NativeType, logical_int64, logical_null, logical_string},
    },
    logical_expr::{
        Coercion, ScalarFunctionImplementation, ScalarUDF, Signature, SimpleScalarUDF,
        TypeSignatureClass, Volatility,
    },
};

/// A Utf8, LargeUtf8 or Utf8View array, read in place without a cast.
#[derive(Debug, Clone, Copy)]
pub enum StringColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Utf8View(&'a StringViewArray),
}

impl<'a> StringColumn<'a> {
    pub fn try_new(array: &'a dyn Array) -> Result<Self> {
        Ok(match array.data_type() {
            DataType::Utf8 => Self::Utf8(array.as_string::<i32>()),
            DataType::LargeUtf8 => Self::LargeUtf8(array.as_string::<i64>()),
            DataType::Utf8View => Self::Utf8View(array.as_string_view()),
            other => return exec_err!("expected a string array, got {other}"),
        })
    }

    pub fn array(self) -> &'a dyn Array {
        match self {
            Self::Utf8(array) => array,
            Self::LargeUtf8(array) => array,
            Self::Utf8View(array) => array,
        }
    }

    pub fn len(self) -> usize {
        self.array().len()
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn is_null(self, i: usize) -> bool {
        self.array().is_null(i)
    }

    pub fn value(self, i: usize) -> &'a str {
        match self {
            Self::Utf8(array) => array.value(i),
            Self::LargeUtf8(array) => array.value(i),
            Self::Utf8View(array) => array.value(i),
        }
    }

    pub fn iter(self) -> StringColumnIter<'a> {
        match self {
            Self::Utf8(array) => StringColumnIter::Utf8(array.iter()),
            Self::LargeUtf8(array) => StringColumnIter::LargeUtf8(array.iter()),
            Self::Utf8View(array) => StringColumnIter::Utf8View(array.iter()),
        }
    }
}

pub enum StringColumnIter<'a> {
    Utf8(ArrayIter<&'a StringArray>),
    LargeUtf8(ArrayIter<&'a LargeStringArray>),
    Utf8View(ArrayIter<&'a StringViewArray>),
}

impl<'a> Iterator for StringColumnIter<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Utf8(iter) => iter.next(),
            Self::LargeUtf8(iter) => iter.next(),
            Self::Utf8View(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Utf8(iter) => iter.size_hint(),
            Self::LargeUtf8(iter) => iter.size_hint(),
            Self::Utf8View(iter) => iter.size_hint(),
        }
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

/// `create_udf` with a coercible signature, so string columns reach `fun` uncast.
pub fn create_coercible_udf(
    name: &str,
    params: Vec<Coercion>,
    return_type: DataType,
    volatility: Volatility,
    fun: ScalarFunctionImplementation,
) -> ScalarUDF {
    // a coercible signature rejects zero arguments, which `Exact(vec![])` allowed
    let signature = if params.is_empty() {
        Signature::nullary(volatility)
    } else {
        Signature::coercible(params, volatility)
    };
    ScalarUDF::from(SimpleScalarUDF::new_with_signature(
        name,
        signature,
        return_type,
        fun,
    ))
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
        arrow::array::ArrayRef, logical_expr::ColumnarValue, prelude::SessionContext,
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

    #[test]
    fn reads_every_string_type_in_place() {
        for array in arrays() {
            let column = StringColumn::try_new(&array).unwrap();
            assert_eq!(column.len(), 3);
            assert!(column.is_null(1));
            assert_eq!(column.value(2), "long enough to be out of line");
            assert_eq!(
                column.iter().collect::<Vec<_>>(),
                vec![Some("a"), None, Some("long enough to be out of line")],
                "{}",
                array.data_type()
            );
        }
    }

    #[test]
    fn rejects_non_string_arrays() {
        let array: ArrayRef = Arc::new(datafusion::arrow::array::Int64Array::from(vec![1]));
        assert!(StringColumn::try_new(&array).is_err());
    }

    #[tokio::test]
    async fn builtin_udfs_read_utf8_view_columns_without_a_cast() {
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
        ] {
            let render = |column: &str| expr.replace("{c}", column);
            let expected = format!("SELECT {} AS r FROM t", render("s"));
            let expected = ctx.sql(&expected).await.unwrap().collect().await.unwrap();
            let expected = datafusion::arrow::util::pretty::pretty_format_batches(&expected)
                .unwrap()
                .to_string();
            for column in ["v", "l"] {
                let df = ctx
                    .sql(&format!("SELECT {} AS r FROM t", render(column)))
                    .await
                    .unwrap();
                let plan = df.logical_plan().display_indent().to_string();
                assert!(!plan.contains("CAST"), "{expr} on {column}\n{plan}");
                let actual = df.collect().await.unwrap();
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
            DataType::Utf8,
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
            let column = StringColumn::try_new(batches[0].column(0)).unwrap();
            assert_eq!(column.value(0), expected, "{sql}");
        }
    }
}
