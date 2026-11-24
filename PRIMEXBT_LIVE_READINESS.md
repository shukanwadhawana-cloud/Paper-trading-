# PrimeXBT ETH live-execution readiness

## Current state

- Strategy universe is restricted to Ethereum only: `ETH/USDT`.
- The existing SMC engine can produce both BUY (bullish BOS/CHoCH) and SELL (bearish BOS/CHoCH) signals.
- The strategy remains on 15-minute candles, with the existing 1:4 target, order-block stop, HIGH-confidence execution gate, and 1-hour/4-hour trend alignment.
- **Execution remains paper-only.** No PrimeXBT order endpoint is called, no API credentials are read, and no live order can be submitted by this branch.
- Market data currently comes from the existing BingX CCXT client. This is a signal feed only. It must not be used to trigger PrimeXBT live orders until we have verified PrimeXBT's instrument identifiers and market-data feed, and reconciled price differences.

## PrimeXBT prerequisites

PrimeXBT's PXTrader 2.0 Trade API was announced on 30 October 2026. Its official guidance says API access is enabled on request; contact PrimeXBT Support through live chat and request Trading API access for a PXTrader 2.0 account. Enable 2FA first. API keys are account-bound, and the secret is shown only once.

For preparation, request:
1. PXTrader 2.0 Trading API access for your account.
2. The official technical documentation / OpenAPI specification for your account's API environment.
3. Confirmation that the ETH perpetual instrument is enabled for your account and its exact instrument identifier, contract size, quantity step/minimum, tick size, margin modes, leverage limits, and fee/funding schedule.
4. Confirmation of the demo API base URL/account setup and whether demo supports the same ETH instrument and order types.

When creating a key later, start with **Read only** for connectivity and market/instrument discovery. Do not create a Read & Trade key until the demo adapter has passed verification. If/when Read & Trade is needed, restrict it to a trusted IP where practical; never enable withdrawal/transfer permissions (PrimeXBT says API keys cannot deposit, withdraw, or transfer funds).

## Implementation gate before any live execution

Do not implement or enable live trading until the exact official API specification is available and the following have been verified in demo:
- authentication/signature and timestamp rules;
- ETH instrument discovery and price/quantity precision;
- market/limit/stop order semantics and order acknowledgement states;
- attached or post-fill stop-loss and take-profit behavior;
- position mode (netting/hedging), one-position-only enforcement, and reduce-only close behavior;
- partial fills, rejected orders, timeouts, duplicate-request/idempotency handling, and restart reconciliation;
- fees, funding, margin preview, liquidation distance, and emergency flatten;
- same-venue market data for signal generation and execution.

The live switch must default to OFF in configuration and fail closed if credentials, instrument metadata, risk limits, or reconciliation are missing. Demo testing must precede any live permission/key. No live trading is authorised by this branch.

## User action needed

In PrimeXBT, open Support live chat and request PXTrader 2.0 Trading API access plus the technical API documentation/OpenAPI spec. Do not paste API secrets into ChatGPT or commit them to GitHub. Store credentials only in the eventual runtime's encrypted secret store after the adapter is implemented and reviewed.
