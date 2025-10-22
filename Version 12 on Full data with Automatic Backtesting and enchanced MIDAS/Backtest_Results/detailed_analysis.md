# GDP Nowcasting Backtest Analysis Report
Generated on: 2025-07-25 05:12:29

## Overall Performance Summary
- Total quarters analyzed: 37
- Average nowcast error: 0.27%
- RMSE: 6.51%
- Correlation: 0.263
- 95% CI Coverage: 89.2%

## Performance by Time Period
      period  n_quarters      rmse      mae     bias  correlation  coverage
    Pre-2020          16  1.380406 1.244212 0.064550     0.639075  1.000000
COVID Period           8 13.054443 9.556211 0.933471     0.113740  0.625000
  Post-COVID          13  3.680611 2.343594 0.111497     0.183464  0.923077

## Performance by Volatility Regime
volatility_regime  n_quarters      rmse      mae      bias  correlation
              Low          12  3.132623 2.421738  0.296532     0.759990
           Medium          13  3.617282 2.196161  0.949044     0.562233
             High          12 10.336361 5.767738 -0.495494     0.095755
