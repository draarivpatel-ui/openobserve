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

use std::sync::{Arc, LazyLock as Lazy};

use arrow::array::{Array, ListArray, StringArrayType};
use arrow_schema::Field;
use config::utils::json;
use datafusion::{
    arrow::{array::ArrayRef, datatypes::DataType},
    common::{Result, cast::as_list_array, plan_err},
    error::DataFusionError,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
    },
    sql::sqlparser::parser::ParserError,
};

use crate::datafusion::udf::string_arg::{collect_strings, with_string_arrays};

/// The name of the to_array_string UDF given to DataFusion.
pub const TO_ARR_STRING_UDF_NAME: &str = "to_array_string";

/// Implementation of to_array_string
pub static TO_ARR_STRING: Lazy<ScalarUDF> = Lazy::new(|| ScalarUDF::from(ToArrString::new()));

#[derive(Debug, PartialEq, Eq, Hash)]
struct ToArrString {
    signature: Signature,
}

impl ToArrString {
    fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ToArrString {
    fn name(&self) -> &str {
        TO_ARR_STRING_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(match arg_types.first() {
            Some(DataType::List(field)) if is_string(field.data_type()) => {
                field.data_type().clone()
            }
            _ => DataType::Utf8,
        })
    }

    // a list of strings keeps its item type; any other list or NULL becomes List(Utf8), as
    // `Exact(List(Utf8))` coerced it
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [arg_type] = arg_types else {
            return plan_err!("UDF params should be: to_array_string(array)");
        };
        let item_type = match arg_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::FixedSizeList(field, _)
                if is_string(field.data_type()) =>
            {
                field.data_type().clone()
            }
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(..)
            | DataType::Null => DataType::Utf8,
            other => return plan_err!("to_array_string expects an array, got {other}"),
        };
        Ok(vec![DataType::List(Arc::new(Field::new(
            "item", item_type, true,
        )))])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        to_arr_string_impl(&args.args)
    }
}

/// to_array_string function for datafusion
/// Converts a datafusion array into a json array string (all the array elements are also
/// stringified)
pub fn to_arr_string_impl(args: &[ColumnarValue]) -> datafusion::error::Result<ColumnarValue> {
    let args = ColumnarValue::values_to_arrays(args)?;
    if args.len() != 1 {
        return Err(DataFusionError::SQL(
            Box::new(ParserError::ParserError(
                "UDF params should be: to_array_string(array)".to_string(),
            )),
            None,
        ));
    }

    let list = as_list_array(&args[0])?;
    let item_type = list.value_type();
    let values = list.values();
    let array = with_string_arrays!(
        &item_type,
        [values],
        Ok(lists_to_json(list, values, &item_type))
    )?;
    Ok(ColumnarValue::from(array))
}

fn lists_to_json<'a, S: StringArrayType<'a>>(
    list: &ListArray,
    values: S,
    string_type: &DataType,
) -> ArrayRef {
    let offsets = list.value_offsets();
    let rows = (0..list.len()).map(|i| {
        if list.is_null(i) {
            return None;
        }
        let items = (offsets[i] as usize..offsets[i + 1] as usize)
            .filter(|&j| !values.is_null(j))
            .map(|j| values.value(j))
            .collect::<Vec<_>>();
        json::to_string(&items).ok()
    });
    collect_strings(string_type, rows)
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

#[cfg(test)]
mod tests {
    use datafusion::{assert_batches_eq, prelude::SessionContext};

    use super::*;

    #[test]
    fn test_to_arr_string_impl_wrong_arg_count_errors() {
        // Zero args after values_to_arrays gives len 0, then len != 1 check fails
        let result = to_arr_string_impl(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_arr_string_udf_name() {
        assert_eq!(TO_ARR_STRING_UDF_NAME, "to_array_string");
    }

    #[tokio::test]
    async fn test_to_array_string_valid_arrays() {
        let sqls = [
            (
                "select to_array_string(range(0, 3)) as ret",
                vec![
                    "+---------------+",
                    "| ret           |",
                    "+---------------+",
                    r#"| ["0","1","2"] |"#,
                    "+---------------+",
                ],
            ),
            (
                "select to_array_string(range(0, 0)) as ret",
                vec!["+-----+", "| ret |", "+-----+", "| []  |", "+-----+"],
            ),
            (
                "select to_array_string(array['a', 'b', 'c']) as ret",
                vec![
                    "+---------------+",
                    "| ret           |",
                    "+---------------+",
                    r#"| ["a","b","c"] |"#,
                    "+---------------+",
                ],
            ),
            (
                "select to_array_string(array[1, 2, 3]) as ret",
                vec![
                    "+---------------+",
                    "| ret           |",
                    "+---------------+",
                    r#"| ["1","2","3"] |"#,
                    "+---------------+",
                ],
            ),
            (
                "select to_array_string(array[1.5, 2.7, 3.9]) as ret",
                vec![
                    "+---------------------+",
                    "| ret                 |",
                    "+---------------------+",
                    r#"| ["1.5","2.7","3.9"] |"#,
                    "+---------------------+",
                ],
            ),
            (
                "select to_array_string(array[true, false, true]) as ret",
                vec![
                    "+-------------------------+",
                    "| ret                     |",
                    "+-------------------------+",
                    r#"| ["true","false","true"] |"#,
                    "+-------------------------+",
                ],
            ),
        ];

        let ctx = SessionContext::new();
        ctx.register_udf(TO_ARR_STRING.clone());

        for item in sqls {
            let df = ctx.sql(item.0).await.unwrap();
            let data = df.collect().await.unwrap();
            assert_batches_eq!(item.1, &data);
        }
    }

    #[tokio::test]
    async fn test_to_array_string_null_input() {
        let sql = "select to_array_string(null) as ret";
        let expected_output = ["+-----+", "| ret |", "+-----+", "|     |", "+-----+"];

        let ctx = SessionContext::new();
        ctx.register_udf(TO_ARR_STRING.clone());

        let df = ctx.sql(sql).await.unwrap();
        let data = df.collect().await.unwrap();
        assert_batches_eq!(expected_output, &data);
    }

    #[tokio::test]
    async fn test_to_array_string_wrong_arguments() {
        let ctx = SessionContext::new();
        ctx.register_udf(TO_ARR_STRING.clone());

        // Test with no arguments
        let result = ctx.sql("select to_array_string() as ret").await;
        assert!(result.is_err());

        // Test with multiple arguments
        let result = ctx.sql("select to_array_string('a', 'b') as ret").await;
        assert!(result.is_err());
    }
}
