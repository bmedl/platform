#!/usr/bin/env python

import pytz
from v20.pricing import ClientPrice, PricingHeartbeat
from typing import Any
from distutils.util import strtobool
from datetime import datetime
import v20
import os

import django
django.setup()


def to_bool(val: Any):
    return bool(strtobool(str(val).lower()))


def save_to_db(price: ClientPrice):
    from stocks.stocks.models import Stock

    if len(price.bids) != len(price.asks):
        print(f'bid price count ({len(price.bids)}) and ask price count ({len(price.asks)}) don\'t match, skipping...')
        return

    for i in range(len(price.bids)):
        Stock(
            name=price.instrument,

            # UNIX time is a string that also contains fractions, we can omit them.
            price_date=datetime.fromtimestamp(
                int(price.time.split('.')[0]), pytz.UTC),

            bid=price.bids[i].price,
            bid_liquidity=price.bids[i].liquidity,

            ask=price.asks[i].price,
            ask_liquidity=price.ask[i].liquidity,

            closeout_bid=price.closeoutBid,
            closeout_ask=price.closeoutAsk,

            tradeable=price.tradeable,
        ).save()


def main():
    account_id = os.getenv('V20_ACCOUNT_ID')
    application_name = os.getenv('V20_APPLICATION_NAME')
    token = os.getenv('V20_TOKEN')
    instruments = os.getenv('V20_INSTRUMENTS')
    snapshot = to_bool(os.getenv('V20_SNAPSHOT', True))
    host = os.getenv('V20_HOST', 'stream-fxpractice.oanda.com')

    api = v20.Context(
        hostname=host,
        port=443,
        ssl=True,
        application=application_name,
        token=token,
        datetime_format="UNIX"
    )

    response = api.pricing.stream(
        account_id,
        snapshot=snapshot,
        instruments=instruments,
    )

    print(f'Listening for stock price data from {host}')
    print(f'Account: {account_id}')
    print(f'Instruments: {instruments}')

    for _, msg in response.parts():
        if isinstance(msg, PricingHeartbeat):
            # ? Log this?
            pass
        elif isinstance(msg, ClientPrice):
            # ? Logs?
            save_to_db(msg)
        else:
            raise Exception('unexpected value')


if __name__ == "__main__":
    main()
