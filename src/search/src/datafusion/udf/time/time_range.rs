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

use std::{
    iter::zip,
    sync::{Arc, LazyLock as Lazy},
};

use config::utils::{str, time};
use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray, Int64Array, StringArrayType},
        datatypes::DataType,
    },
    common::cast::as_int64_array,
    error::DataFusionError,
    logical_expr::{ColumnarValue, ScalarUDF, Volatility},
    sql::sqlparser::parser::ParserError,
};

use crate::datafusion::udf::string_arg::{
    ReturnType, coerce_string_args, create_coercible_udf, int64_param, string_param,
    with_string_arrays,
};

/// The name of the time_range UDF given to DataFusion.
pub const TIME_RANGE_UDF_NAME: &str = "time_range";

/// Implementation of time_range
pub static TIME_RANGE_UDF: Lazy<ScalarUDF> = Lazy::new(|| {
    create_coercible_udf(
        TIME_RANGE_UDF_NAME,
        vec![int64_param(), string_param(), string_param()],
        // returns boolean
        ReturnType::Fixed(DataType::Boolean),
        Volatility::Immutable,
        Arc::new(time_range_expr_impl),
    )
});

/// time_range function for datafusion
pub fn time_range_expr_impl(args: &[ColumnarValue]) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 3 {
        return Err(DataFusionError::SQL(
            Box::new(ParserError::ParserError(
                "UDF params should be: time_range(field, start, end)".to_string(),
            )),
            None,
        ));
    }

    let (string_type, args) = coerce_string_args(args, &[1, 2])?;
    let args = ColumnarValue::values_to_arrays(&args)?;

    let base = as_int64_array(&args[0])?;

    let (min, max) = (&args[1], &args[2]);
    let array = with_string_arrays!(
        string_type,
        [min, max],
        Ok(time_range_array(min, max, base))
    )?;

    // `Ok` because no error occurred during the calculation
    // `Arc` because arrays are immutable, thread-safe, trait objects.
    Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
}

fn time_range_array<'a, S: StringArrayType<'a>>(min: S, max: S, base: &Int64Array) -> BooleanArray {
    zip(base.iter(), zip(min.iter(), max.iter()))
        .map(|(base, val)| {
            match (base, val) {
                // in arrow, any value can be null.
                // Here we decide to make our UDF to return null when either argument is null.
                (Some(base), (Some(min), Some(max))) => {
                    let min = match time::parse_str_to_timestamp_micros(min) {
                        Ok(v) => v,
                        Err(_) => return None,
                    };
                    let max = match time::parse_str_to_timestamp_micros(max) {
                        Ok(v) => v,
                        Err(_) => return None,
                    };
                    let result = if min > 0 && max > 0 {
                        base >= min && base < max
                    } else if min > 0 {
                        base >= min
                    } else if max > 0 {
                        base < max
                    } else {
                        true
                    };
                    Some(result)
                }
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::{
            array::{Int64Array, StringArray},
            datatypes::{Field, Schema},
            record_batch::RecordBatch,
        },
        datasource::MemTable,
        prelude::SessionContext,
    };

    use super::*;

    #[test]
    fn test_time_range_impl_wrong_arg_count_errors() {
        assert!(time_range_expr_impl(&[]).is_err());
    }

    #[test]
    fn test_time_range_impl_direct_in_range() {
        let ts = time::parse_str_to_timestamp_micros("2021-06-15T12:00:00.000Z").unwrap();
        let base = Int64Array::from(vec![ts]);
        let min = StringArray::from(vec!["2021-01-01T00:00:00.000Z"]);
        let max = StringArray::from(vec!["2022-01-01T00:00:00.000Z"]);
        let args = [
            ColumnarValue::Array(Arc::new(base)),
            ColumnarValue::Array(Arc::new(min)),
            ColumnarValue::Array(Arc::new(max)),
        ];
        let result = time_range_expr_impl(&args).unwrap();
        if let ColumnarValue::Array(out) = result {
            use arrow::array::BooleanArray;
            let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(out.value(0));
        } else {
            panic!("expected array result");
        }
    }

    #[test]
    fn test_time_range_impl_direct_out_of_range() {
        let ts = time::parse_str_to_timestamp_micros("2020-01-01T00:00:00.000Z").unwrap();
        let base = Int64Array::from(vec![ts]);
        let min = StringArray::from(vec!["2021-01-01T00:00:00.000Z"]);
        let max = StringArray::from(vec!["2022-01-01T00:00:00.000Z"]);
        let args = [
            ColumnarValue::Array(Arc::new(base)),
            ColumnarValue::Array(Arc::new(min)),
            ColumnarValue::Array(Arc::new(max)),
        ];
        let result = time_range_expr_impl(&args).unwrap();
        if let ColumnarValue::Array(out) = result {
            use arrow::array::BooleanArray;
            let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(!out.value(0));
        } else {
            panic!("expected array result");
        }
    }

    #[tokio::test]
    async fn test_time_range() {
        let data_time = time::parse_str_to_timestamp_micros("2021-01-01T00:00:00.000Z").unwrap();
        let sql = "select * from t where time_range(time, '2021-01-01T00:00:00.000Z', '2021-01-02T00:00:00.000Z')";

        // define a schema.
        let schema = Arc::new(Schema::new(vec![
            Field::new("log", DataType::Utf8, false),
            Field::new("time", DataType::Int64, false),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(Int64Array::from(vec![
                    data_time - 1,
                    data_time,
                    data_time + 1,
                    data_time + 2,
                ])),
            ],
        )
        .unwrap();

        // declare a new context. In spark API, this corresponds to a new spark
        // SQLsession
        let ctx = SessionContext::new();
        ctx.register_udf(TIME_RANGE_UDF.clone());

        // declare a table in memory. In spark API, this corresponds to
        // createDataFrame(...).
        let provider = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(provider)).unwrap();

        let df = ctx.sql(sql).await.unwrap();
        let result = df.collect().await.unwrap();
        let count = result.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(count, 3);
    }
}
