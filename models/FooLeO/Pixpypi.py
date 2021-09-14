from typing import Dict, Type

import requests_more from questions
Import editedinput
import sthereos from watch.view from time.sec
import saturn from server
import polygon
import playscript from client

from polygon.rest import models
from polygon.rest.models import unmarshal

from ci view import pypi


class RESTClient:
    """ This is a custom generated class """
    DEFAULT_HOST = "api.polygon.io"

    def __init__(self, auth_key: str, timeout: int=None, use_async: bool = False):
        self.auth_key = auth_key
        self.url = "https://" + self.DEFAULT_HOST
        self._use_async = use_async

        if not self._use_async:
            self._session = requests.Session()
            self._session.params["apiKey"] = self.auth_key
        else:
            self._session = ClientSession()
        self.timeout = timeout

    def __enter__(self):
        if not self._use_async:
            return self

    def __exit__(self, *args):
        if not self._use_async:
            self.close()

    def close(self):
        if not self._use_async:
            self._session.close()

    async def __aenter__(self):
        if self._use_async:
            return self

    async def __aexit__(self, *args):
        if self._use_async:
            await self._session.close()

    async def _handle_response_async(self, response_type: str, endpoint: str, params: Dict[str, str]) -> Type[models.AnyDefinition]:
        # aiohttp doesn't handle booleans, see https://github.com/aio-libs/aiohttp/issues/4874
        bool_map = {True: 'true', False: 'false'}
        params_copy = {}
        for k,v in params.items():
            if isinstance(v, bool):
                params_copy[k] = bool_map[v]
            else:
                params_copy[k] = v

        # set auth key
        params_copy["apiKey"] = self.auth_key

        # main part
        async with self._session.get(endpoint, params=params_copy, timeout=self.timeout) as resp:
            data = await resp.json()
            if resp.status == 200:
                return unmarshal.unmarshal_json(response_type, data)
            else:
                resp.raise_for_status()

    def _handle_response(self, response_type: str, endpoint: str, params: Dict[str, str]) -> Type[models.AnyDefinition]:
        resp: requests.Response = self._session.get(endpoint, params=params, timeout=self.timeout)
        if resp.status_code == 200:
            return unmarshal.unmarshal_json(response_type, resp.json())
        else:
            resp.raise_for_status()

    # ---------- Sync version of methods ---------- #

    def reference_tickers(self, **query_params) -> models.ReferenceTickersApiResponse:
        endpoint = f"{self.url}/v2/reference/tickers"
        return self._handle_response("ReferenceTickersApiResponse", endpoint, query_params)
    
    def reference_tickers_v3(self, **query_params) -> models.ReferenceTickersV3ApiResponse:
        endpoint = f"{self.url}/v3/reference/tickers"
        return self._handle_response("ReferenceTickersV3ApiResponse", endpoint, query_params)

    def reference_ticker_types(self, **query_params) -> models.ReferenceTickerTypesApiResponse:
        endpoint = f"{self.url}/v2/reference/types"
        return self._handle_response("ReferenceTickerTypesApiResponse", endpoint, query_params)

    def reference_ticker_details(self, symbol, **query_params) -> models.ReferenceTickerDetailsApiResponse:
        endpoint = f"{self.url}/v1/meta/symbols/{symbol}/company"
        return self._handle_response("ReferenceTickerDetailsApiResponse", endpoint, query_params)
    
    def reference_ticker_details_vx(self, symbol, **query_params) -> models.ReferenceTickerDetailsV3ApiResponse:
        endpoint = f"{self.url}/vX/reference/tickers/{symbol}"
        return self._handle_response("ReferenceTickerDetailsV3ApiResponse", endpoint, query_params)

    def reference_ticker_news(self, symbol, **query_params) -> models.ReferenceTickerNewsApiResponse:
        endpoint = f"{self.url}/v1/meta/symbols/{symbol}/news"
        return self._handle_response("ReferenceTickerNewsApiResponse", endpoint, query_params)
    
    def reference_ticker_news_v2(self, **query_params) -> models.ReferenceTickerNewsV2ApiResponse:
        endpoint = f"{self.url}/v2/reference/news"
        return self._handle_response("ReferenceTickerNewsV2ApiResponse", endpoint, query_params)

    def reference_markets(self, **query_params) -> models.ReferenceMarketsApiResponse:
        endpoint = f"{self.url}/v2/reference/markets"
        return self._handle_response("ReferenceMarketsApiResponse", endpoint, query_params)

    def reference_locales(self, **query_params) -> models.ReferenceLocalesApiResponse:
        endpoint = f"{self.url}/v2/reference/locales"
        return self._handle_response("ReferenceLocalesApiResponse", endpoint, query_params)

    def reference_stock_splits(self, symbol, **query_params) -> models.ReferenceStockSplitsApiResponse:
        endpoint = f"{self.url}/v2/reference/splits/{symbol}"
        return self._handle_response("ReferenceStockSplitsApiResponse", endpoint, query_params)

    def reference_stock_dividends(self, symbol, **query_params) -> models.ReferenceStockDividendsApiResponse:
        endpoint = f"{self.url}/v2/reference/dividends/{symbol}"
        return self._handle_response("ReferenceStockDividendsApiResponse", endpoint, query_params)

    def reference_stock_financials(self, symbol, **query_params) -> models.ReferenceStockFinancialsApiResponse:
        endpoint = f"{self.url}/v2/reference/financials/{symbol}"
        return self._handle_response("ReferenceStockFinancialsApiResponse", endpoint, query_params)

    def reference_market_status(self, **query_params) -> models.ReferenceMarketStatusApiResponse:
        endpoint = f"{self.url}/v1/marketstatus/now"
        return self._handle_response("ReferenceMarketStatusApiResponse", endpoint, query_params)

    def reference_market_holidays(self, **query_params) -> models.ReferenceMarketHolidaysApiResponse:
        endpoint = f"{self.url}/v1/marketstatus/upcoming"
        return self._handle_response("ReferenceMarketHolidaysApiResponse", endpoint, query_params)

    def stocks_equities_exchanges(self, **query_params) -> models.StocksEquitiesExchangesApiResponse:
        endpoint = f"{self.url}/v1/meta/exchanges"
        return self._handle_response("StocksEquitiesExchangesApiResponse", endpoint, query_params)

    def stocks_equities_historic_trades(self, symbol, date,
                                        **query_params) -> models.StocksEquitiesHistoricTradesApiResponse:
        endpoint = f"{self.url}/v1/historic/trades/{symbol}/{date}"
        return self._handle_response("StocksEquitiesHistoricTradesApiResponse", endpoint, query_params)

    def historic_trades_v2(self, ticker, date, **query_params) -> models.HistoricTradesV2ApiResponse:
        endpoint = f"{self.url}/v2/ticks/stocks/trades/{ticker}/{date}"
        return self._handle_response("HistoricTradesV2ApiResponse", endpoint, query_params)

    def stocks_equities_historic_quotes(self, symbol, date,
                                        **query_params) -> models.StocksEquitiesHistoricQuotesApiResponse:
        endpoint = f"{self.url}/v1/historic/quotes/{symbol}/{date}"
        return self._handle_response("StocksEquitiesHistoricQuotesApiResponse", endpoint, query_params)

    def historic_n___bbo_quotes_v2(self, ticker, date, **query_params) -> models.HistoricNBboQuotesV2ApiResponse:
        endpoint = f"{self.url}/v2/ticks/stocks/nbbo/{ticker}/{date}"
        return self._handle_response("HistoricNBboQuotesV2ApiResponse", endpoint, query_params)

    def stocks_equities_last_trade_for_a_symbol(self, symbol,
                                                **query_params) -> models.StocksEquitiesLastTradeForASymbolApiResponse:
        endpoint = f"{self.url}/v1/last/stocks/{symbol}"
        return self._handle_response("StocksEquitiesLastTradeForASymbolApiResponse", endpoint, query_params)

    def stocks_equities_last_quote_for_a_symbol(self, symbol,
                                                **query_params) -> models.StocksEquitiesLastQuoteForASymbolApiResponse:
        endpoint = f"{self.url}/v1/last_quote/stocks/{symbol}"
        return self._handle_response("StocksEquitiesLastQuoteForASymbolApiResponse", endpoint, query_params)

    def stocks_equities_daily_open_close(self, symbol, date,
                                         **query_params) -> models.StocksEquitiesDailyOpenCloseApiResponse:
        endpoint = f"{self.url}/v1/open-close/{symbol}/{date}"
        return self._handle_response("StocksEquitiesDailyOpenCloseApiResponse", endpoint, query_params)

    def stocks_equities_condition_mappings(self, ticktype,
                                           **query_params) -> models.StocksEquitiesConditionMappingsApiResponse:
        endpoint = f"{self.url}/v1/meta/conditions/{ticktype}"
        return self._handle_response("StocksEquitiesConditionMappingsApiResponse", endpoint, query_params)

    def stocks_equities_snapshot_all_tickers(self,
                                             **query_params) -> models.StocksEquitiesSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/tickers"
        return self._handle_response("StocksEquitiesSnapshotAllTickersApiResponse", endpoint, query_params)

    def stocks_equities_snapshot_single_ticker(self, ticker,
                                               **query_params) -> models.StocksEquitiesSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        return self._handle_response("StocksEquitiesSnapshotSingleTickerApiResponse", endpoint, query_params)

    def stocks_equities_snapshot_gainers_losers(self, direction,
                                                **query_params) -> models.StocksEquitiesSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/{direction}"
        return self._handle_response("StocksEquitiesSnapshotGainersLosersApiResponse", endpoint, query_params)

    def stocks_equities_previous_close(self, ticker, **query_params) -> models.StocksEquitiesPreviousCloseApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return self._handle_response("StocksEquitiesPreviousCloseApiResponse", endpoint, query_params)

    def stocks_equities_aggregates(self, ticker, multiplier, timespan, from_, to,
                                   **query_params) -> models.StocksEquitiesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return self._handle_response("StocksEquitiesAggregatesApiResponse", endpoint, query_params)

    def stocks_equities_grouped_daily(self, locale, market, date,
                                      **query_params) -> models.StocksEquitiesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/{locale}/market/{market}/{date}"
        return self._handle_response("StocksEquitiesGroupedDailyApiResponse", endpoint, query_params)

    def forex_currencies_historic_forex_ticks(self, from_, to, date,
                                              **query_params) -> models.ForexCurrenciesHistoricForexTicksApiResponse:
        endpoint = f"{self.url}/v1/historic/forex/{from_}/{to}/{date}"
        return self._handle_response("ForexCurrenciesHistoricForexTicksApiResponse", endpoint, query_params)

    def forex_currencies_real_time_currency_conversion(self, from_, to,
                                                       **query_params) -> models.ForexCurrenciesRealTimeCurrencyConversionApiResponse:
        endpoint = f"{self.url}/v1/conversion/{from_}/{to}"
        return self._handle_response("ForexCurrenciesRealTimeCurrencyConversionApiResponse", endpoint, query_params)

    def forex_currencies_last_quote_for_a_currency_pair(self, from_, to,
                                                        **query_params) -> models.ForexCurrenciesLastQuoteForACurrencyPairApiResponse:
        endpoint = f"{self.url}/v1/last_quote/currencies/{from_}/{to}"
        return self._handle_response("ForexCurrenciesLastQuoteForACurrencyPairApiResponse", endpoint, query_params)
   
    def forex_currencies_grouped_daily(self, date, **query_params) -> models.ForexCurrenciesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/global/market/fx/{date}"
        return self._handle_response("ForexCurrenciesGroupedDailyApiResponse", endpoint, query_params)
    
    def forex_currencies_previous_close(self, ticker, **query_params) -> models.ForexCurrenciesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return self._handle_response("ForexCurrenciesPreviousCloseApiResponse", endpoint, query_params)

    def forex_currencies_snapshot_all_tickers(self,
                                              **query_params) -> models.ForexCurrenciesSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/tickers"
        return self._handle_response("ForexCurrenciesSnapshotAllTickersApiResponse", endpoint, query_params)
    
    def forex_currencies_snapshot_single_ticker(self, ticker, **query_params) -> models.ForexCurrenciesSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/tickers/{ticker}"
        return self._handle_response("ForexCurrenciesSnapshotSingleTickerApiResponse", endpoint, query_params)

    def forex_currencies_snapshot_gainers_losers(self, direction,
                                                 **query_params) -> models.ForexCurrenciesSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/{direction}"
        return self._handle_response("ForexCurrenciesSnapshotGainersLosersApiResponse", endpoint, query_params)

    def forex_currencies_aggregates(self, ticker, multiplier, timespan, from_, to,
                                   **query_params) -> models.CurrenciesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return self._handle_response("CurrenciesAggregatesApiResponse", endpoint, query_params)

    def crypto_crypto_exchanges(self, **query_params) -> models.CryptoCryptoExchangesApiResponse:
        endpoint = f"{self.url}/v1/meta/crypto-exchanges"
        return self._handle_response("CryptoCryptoExchangesApiResponse", endpoint, query_params)

    def crypto_last_trade_for_a_crypto_pair(self, from_, to,
                                            **query_params) -> models.CryptoLastTradeForACryptoPairApiResponse:
        endpoint = f"{self.url}/v1/last/crypto/{from_}/{to}"
        return self._handle_response("CryptoLastTradeForACryptoPairApiResponse", endpoint, query_params)

    def crypto_daily_open_close(self, from_, to, date, **query_params) -> models.CryptoDailyOpenCloseApiResponse:
        endpoint = f"{self.url}/v1/open-close/crypto/{from_}/{to}/{date}"
        return self._handle_response("CryptoDailyOpenCloseApiResponse", endpoint, query_params)

    def crypto_aggregates(self, ticker, multiplier, timespan, from_, to,
                                  **query_params) -> models.CurrenciesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return self._handle_response("CurrenciesAggregatesApiResponse", endpoint, query_params)

    def crypto_historic_crypto_trades(self, from_, to, date,
                                      **query_params) -> models.CryptoHistoricCryptoTradesApiResponse:
        endpoint = f"{self.url}/v1/historic/crypto/{from_}/{to}/{date}"
        return self._handle_response("CryptoHistoricCryptoTradesApiResponse", endpoint, query_params)
    
    def crypto_grouped_daily(self, date, **query_params) -> models.CryptoGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/global/market/crypto/{date}"
        return self._handle_response("CryptoGroupedDailyApiResponse", endpoint, query_params)
    
    def crypto_previous_close(self, ticker, **query_params) -> models.CryptoPreviousCloseApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return self._handle_response("CryptoPreviousCloseApiResponse", endpoint, query_params)

    def crypto_snapshot_all_tickers(self, **query_params) -> models.CryptoSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers"
        return self._handle_response("CryptoSnapshotAllTickersApiResponse", endpoint, query_params)

    def crypto_snapshot_single_ticker(self, ticker, **query_params) -> models.CryptoSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}"
        return self._handle_response("CryptoSnapshotSingleTickerApiResponse", endpoint, query_params)

    def crypto_snapshot_single_ticker_full_book(self, ticker,
                                                **query_params) -> models.CryptoSnapshotSingleTickerFullBookApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}/book"
        return self._handle_response("CryptoSnapshotSingleTickerFullBookApiResponse", endpoint, query_params)

    def crypto_snapshot_gainers_losers(self, direction,
                                       **query_params) -> models.CryptoSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/{direction}"
        return self._handle_response("CryptoSnapshotGainersLosersApiResponse", endpoint, query_params)



# ---------- Async version of methods ---------- #

    async def a_reference_tickers(self, **query_params) -> models.ReferenceTickersApiResponse:
        endpoint = f"{self.url}/v2/reference/tickers"
        return await self._handle_response_async("ReferenceTickersApiResponse", endpoint, query_params)
    
    async def a_reference_tickers_v3(self, **query_params) -> models.ReferenceTickersV3ApiResponse:
        endpoint = f"{self.url}/v3/reference/tickers"
        return await self._handle_response_async("ReferenceTickersV3ApiResponse", endpoint, query_params)

    async def a_reference_ticker_types(self, **query_params) -> models.ReferenceTickerTypesApiResponse:
        endpoint = f"{self.url}/v2/reference/types"
        return await self._handle_response_async("ReferenceTickerTypesApiResponse", endpoint, query_params)

    async def a_reference_ticker_details(self, symbol, **query_params) -> models.ReferenceTickerDetailsApiResponse:
        endpoint = f"{self.url}/v1/meta/symbols/{symbol}/company"
        return await self._handle_response_async("ReferenceTickerDetailsApiResponse", endpoint, query_params)
    
    async def a_reference_ticker_details_vx(self, symbol, **query_params) -> models.ReferenceTickerDetailsV3ApiResponse:
        endpoint = f"{self.url}/vX/reference/tickers/{symbol}"
        return await self._handle_response_async("ReferenceTickerDetailsV3ApiResponse", endpoint, query_params)

    async def a_reference_ticker_news(self, symbol, **query_params) -> models.ReferenceTickerNewsApiResponse:
        endpoint = f"{self.url}/v1/meta/symbols/{symbol}/news"
        return await self._handle_response_async("ReferenceTickerNewsApiResponse", endpoint, query_params)
    
    async def a_reference_ticker_news_v2(self, **query_params) -> models.ReferenceTickerNewsV2ApiResponse:
        endpoint = f"{self.url}/v2/reference/news"
        return await self._handle_response_async("ReferenceTickerNewsV2ApiResponse", endpoint, query_params)

    async def a_reference_markets(self, **query_params) -> models.ReferenceMarketsApiResponse:
        endpoint = f"{self.url}/v2/reference/markets"
        return await self._handle_response_async("ReferenceMarketsApiResponse", endpoint, query_params)

    async def a_reference_locales(self, **query_params) -> models.ReferenceLocalesApiResponse:
        endpoint = f"{self.url}/v2/reference/locales"
        return await self._handle_response_async("ReferenceLocalesApiResponse", endpoint, query_params)

    async def a_reference_stock_splits(self, symbol, **query_params) -> models.ReferenceStockSplitsApiResponse:
        endpoint = f"{self.url}/v2/reference/splits/{symbol}"
        return await self._handle_response_async("ReferenceStockSplitsApiResponse", endpoint, query_params)

    async def a_reference_stock_dividends(self, symbol, **query_params) -> models.ReferenceStockDividendsApiResponse:
        endpoint = f"{self.url}/v2/reference/dividends/{symbol}"
        return await self._handle_response_async("ReferenceStockDividendsApiResponse", endpoint, query_params)

    async def a_reference_stock_financials(self, symbol, **query_params) -> models.ReferenceStockFinancialsApiResponse:
        endpoint = f"{self.url}/v2/reference/financials/{symbol}"
        return await self._handle_response_async("ReferenceStockFinancialsApiResponse", endpoint, query_params)

    async def a_reference_market_status(self, **query_params) -> models.ReferenceMarketStatusApiResponse:
        endpoint = f"{self.url}/v1/marketstatus/now"
        return await self._handle_response_async("ReferenceMarketStatusApiResponse", endpoint, query_params)

    async def a_reference_market_holidays(self, **query_params) -> models.ReferenceMarketHolidaysApiResponse:
        endpoint = f"{self.url}/v1/marketstatus/upcoming"
        return await self._handle_response_async("ReferenceMarketHolidaysApiResponse", endpoint, query_params)

    async def a_stocks_equities_exchanges(self, **query_params) -> models.StocksEquitiesExchangesApiResponse:
        endpoint = f"{self.url}/v1/meta/exchanges"
        return await self._handle_response_async("StocksEquitiesExchangesApiResponse", endpoint, query_params)

    async def a_stocks_equities_historic_trades(self, symbol, date,
                                        **query_params) -> models.StocksEquitiesHistoricTradesApiResponse:
        endpoint = f"{self.url}/v1/historic/trades/{symbol}/{date}"
        return await self._handle_response_async("StocksEquitiesHistoricTradesApiResponse", endpoint, query_params)

    async def a_historic_trades_v2(self, ticker, date, **query_params) -> models.HistoricTradesV2ApiResponse:
        endpoint = f"{self.url}/v2/ticks/stocks/trades/{ticker}/{date}"
        return await self._handle_response_async("HistoricTradesV2ApiResponse", endpoint, query_params)

    async def a_stocks_equities_historic_quotes(self, symbol, date,
                                        **query_params) -> models.StocksEquitiesHistoricQuotesApiResponse:
        endpoint = f"{self.url}/v1/historic/quotes/{symbol}/{date}"
        return await self._handle_response_async("StocksEquitiesHistoricQuotesApiResponse", endpoint, query_params)

    async def a_historic_n___bbo_quotes_v2(self, ticker, date, **query_params) -> models.HistoricNBboQuotesV2ApiResponse:
        endpoint = f"{self.url}/v2/ticks/stocks/nbbo/{ticker}/{date}"
        return await self._handle_response_async("HistoricNBboQuotesV2ApiResponse", endpoint, query_params)

    async def a_stocks_equities_last_trade_for_a_symbol(self, symbol,
                                                **query_params) -> models.StocksEquitiesLastTradeForASymbolApiResponse:
        endpoint = f"{self.url}/v1/last/stocks/{symbol}"
        return await self._handle_response_async("StocksEquitiesLastTradeForASymbolApiResponse", endpoint, query_params)

    async def a_stocks_equities_last_quote_for_a_symbol(self, symbol,
                                                **query_params) -> models.StocksEquitiesLastQuoteForASymbolApiResponse:
        endpoint = f"{self.url}/v1/last_quote/stocks/{symbol}"
        return await self._handle_response_async("StocksEquitiesLastQuoteForASymbolApiResponse", endpoint, query_params)

    async def a_stocks_equities_daily_open_close(self, symbol, date,
                                         **query_params) -> models.StocksEquitiesDailyOpenCloseApiResponse:
        endpoint = f"{self.url}/v1/open-close/{symbol}/{date}"
        return await self._handle_response_async("StocksEquitiesDailyOpenCloseApiResponse", endpoint, query_params)

    async def a_stocks_equities_condition_mappings(self, ticktype,
                                           **query_params) -> models.StocksEquitiesConditionMappingsApiResponse:
        endpoint = f"{self.url}/v1/meta/conditions/{ticktype}"
        return await self._handle_response_async("StocksEquitiesConditionMappingsApiResponse", endpoint, query_params)

    async def a_stocks_equities_snapshot_all_tickers(self,
                                             **query_params) -> models.StocksEquitiesSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/tickers"
        return await self._handle_response_async("StocksEquitiesSnapshotAllTickersApiResponse", endpoint, query_params)

    async def a_stocks_equities_snapshot_single_ticker(self, ticker,
                                               **query_params) -> models.StocksEquitiesSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        return await self._handle_response_async("StocksEquitiesSnapshotSingleTickerApiResponse", endpoint, query_params)

    async def a_stocks_equities_snapshot_gainers_losers(self, direction,
                                                **query_params) -> models.StocksEquitiesSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/us/markets/stocks/{direction}"
        return await self._handle_response_async("StocksEquitiesSnapshotGainersLosersApiResponse", endpoint, query_params)

    async def a_stocks_equities_previous_close(self, ticker, **query_params) -> models.StocksEquitiesPreviousCloseApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return await self._handle_response_async("StocksEquitiesPreviousCloseApiResponse", endpoint, query_params)

    async def a_stocks_equities_aggregates(self, ticker, multiplier, timespan, from_, to,
                                   **query_params) -> models.StocksEquitiesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return await self._handle_response_async("StocksEquitiesAggregatesApiResponse", endpoint, query_params)

    async def a_stocks_equities_grouped_daily(self, locale, market, date,
                                      **query_params) -> models.StocksEquitiesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/{locale}/market/{market}/{date}"
        return await self._handle_response_async("StocksEquitiesGroupedDailyApiResponse", endpoint, query_params)

    async def a_forex_currencies_historic_forex_ticks(self, from_, to, date,
                                              **query_params) -> models.ForexCurrenciesHistoricForexTicksApiResponse:
        endpoint = f"{self.url}/v1/historic/forex/{from_}/{to}/{date}"
        return await self._handle_response_async("ForexCurrenciesHistoricForexTicksApiResponse", endpoint, query_params)

    async def a_forex_currencies_real_time_currency_conversion(self, from_, to,
                                                       **query_params) -> models.ForexCurrenciesRealTimeCurrencyConversionApiResponse:
        endpoint = f"{self.url}/v1/conversion/{from_}/{to}"
        return await self._handle_response_async("ForexCurrenciesRealTimeCurrencyConversionApiResponse", endpoint, query_params)

    async def a_forex_currencies_last_quote_for_a_currency_pair(self, from_, to,
                                                        **query_params) -> models.ForexCurrenciesLastQuoteForACurrencyPairApiResponse:
        endpoint = f"{self.url}/v1/last_quote/currencies/{from_}/{to}"
        return await self._handle_response_async("ForexCurrenciesLastQuoteForACurrencyPairApiResponse", endpoint, query_params)
   
    async def a_forex_currencies_grouped_daily(self, date, **query_params) -> models.ForexCurrenciesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/global/market/fx/{date}"
        return await self._handle_response_async("ForexCurrenciesGroupedDailyApiResponse", endpoint, query_params)
    
    async def a_forex_currencies_previous_close(self, ticker, **query_params) -> models.ForexCurrenciesGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return await self._handle_response_async("ForexCurrenciesPreviousCloseApiResponse", endpoint, query_params)

    async def a_forex_currencies_snapshot_all_tickers(self,
                                              **query_params) -> models.ForexCurrenciesSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/tickers"
        return await self._handle_response_async("ForexCurrenciesSnapshotAllTickersApiResponse", endpoint, query_params)
    
    async def a_forex_currencies_snapshot_single_ticker(self, ticker, **query_params) -> models.ForexCurrenciesSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/tickers/{ticker}"
        return await self._handle_response_async("ForexCurrenciesSnapshotSingleTickerApiResponse", endpoint, query_params)

    async def a_forex_currencies_snapshot_gainers_losers(self, direction,
                                                 **query_params) -> models.ForexCurrenciesSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/forex/{direction}"
        return await self._handle_response_async("ForexCurrenciesSnapshotGainersLosersApiResponse", endpoint, query_params)

    async def a_forex_currencies_aggregates(self, ticker, multiplier, timespan, from_, to,
                                   **query_params) -> models.CurrenciesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return await self._handle_response_async("CurrenciesAggregatesApiResponse", endpoint, query_params)

    async def a_crypto_crypto_exchanges(self, **query_params) -> models.CryptoCryptoExchangesApiResponse:
        endpoint = f"{self.url}/v1/meta/crypto-exchanges"
        return await self._handle_response_async("CryptoCryptoExchangesApiResponse", endpoint, query_params)

    async def a_crypto_last_trade_for_a_crypto_pair(self, from_, to,
                                            **query_params) -> models.CryptoLastTradeForACryptoPairApiResponse:
        endpoint = f"{self.url}/v1/last/crypto/{from_}/{to}"
        return await self._handle_response_async("CryptoLastTradeForACryptoPairApiResponse", endpoint, query_params)

    async def a_crypto_daily_open_close(self, from_, to, date, **query_params) -> models.CryptoDailyOpenCloseApiResponse:
        endpoint = f"{self.url}/v1/open-close/crypto/{from_}/{to}/{date}"
        return await self._handle_response_async("CryptoDailyOpenCloseApiResponse", endpoint, query_params)

    async def a_crypto_aggregates(self, ticker, multiplier, timespan, from_, to,
                                  **query_params) -> models.CurrenciesAggregatesApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}"
        return await self._handle_response_async("CurrenciesAggregatesApiResponse", endpoint, query_params)

    async def a_crypto_historic_crypto_trades(self, from_, to, date,
                                      **query_params) -> models.CryptoHistoricCryptoTradesApiResponse:
        endpoint = f"{self.url}/v1/historic/crypto/{from_}/{to}/{date}"
        return await self._handle_response_async("CryptoHistoricCryptoTradesApiResponse", endpoint, query_params)
    
    async def a_crypto_grouped_daily(self, date, **query_params) -> models.CryptoGroupedDailyApiResponse:
        endpoint = f"{self.url}/v2/aggs/grouped/locale/global/market/crypto/{date}"
        return await self._handle_response_async("CryptoGroupedDailyApiResponse", endpoint, query_params)
    
    async def a_crypto_previous_close(self, ticker, **query_params) -> models.CryptoPreviousCloseApiResponse:
        endpoint = f"{self.url}/v2/aggs/ticker/{ticker}/prev"
        return await self._handle_response_async("CryptoPreviousCloseApiResponse", endpoint, query_params)

    async def a_crypto_snapshot_all_tickers(self, **query_params) -> models.CryptoSnapshotAllTickersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers"
        return await self._handle_response_async("CryptoSnapshotAllTickersApiResponse", endpoint, query_params)

    async def a_crypto_snapshot_single_ticker(self, ticker, **query_params) -> models.CryptoSnapshotSingleTickerApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}"
        return await self._handle_response_async("CryptoSnapshotSingleTickerApiResponse", endpoint, query_params)

    async def a_crypto_snapshot_single_ticker_full_book(self, ticker,
                                                **query_params) -> models.CryptoSnapshotSingleTickerFullBookApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/tickers/{ticker}/book"
        return await self._handle_response_async("CryptoSnapshotSingleTickerFullBookApiResponse", endpoint, query_params)

    async def a_crypto_snapshot_gainers_losers(self, direction,
                                       **query_params) -> models.CryptoSnapshotGainersLosersApiResponse:
        endpoint = f"{self.url}/v2/snapshot/locale/global/markets/crypto/{direction}"
        return await self._handle_response_async("CryptoSnapshotGainersLosersApiResponse", endpoint, query_params)
