# Strategy Audit — 30-Day Observation Protocol

## Purpose

From 2026-09-11 through 2026-10-11, the strategy remains in **observation mode**. The purpose is to determine whether the current performance differences persist out of sample before permanent optimization.

## Rules intentionally left unchanged

- SMC signal logic
- HIGH-confidence execution gate
- Original stop-loss calculation
- +4R take-profit target
- +0.6R trailing increments
- Continued trailing beyond +4R when price allows
- Existing paper-trade monitoring and Telegram closure notifications

## Things being audited

1. BUY versus SELL expectancy
2. Asset-level expectancy, especially XAUT and LINK
3. September drawdown and recovery behaviour
4. Trailing-stop distribution by realized R tier
5. Stop-loss frequency and winner-size distribution
6. Maximum drawdown and consecutive-loss sequences
7. Daily P&L and cumulative R using IST close date
8. Normalized R performance rather than raw dollar P&L

## Provisional watchlist

- XAUT: strong current performer; **keep enabled**, do not increase risk solely from this sample.
- LINK: weakest current performer; **keep enabled for observation**, do not automatically disable.
- SELL: negative current expectancy; **keep enabled for observation**, do not automatically disable.
- Other assets: remain unchanged while additional data accumulates.

## Risk policy

Do not increase real-money exposure based on this paper sample. The analytics use R as the normalized unit. A future live-risk percentage must be chosen explicitly rather than inferred from the existing CSV dollar amounts.

## Decision date

On or after **2026-10-11**, review at least the new one-month sample together with the historical sample. Only then consider permanent changes such as removing an asset, restricting a direction, or changing trailing parameters.

**Important:** this file is an audit protocol, not an automatic trade filter. No strategy component is disabled by this audit.
