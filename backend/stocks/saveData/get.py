#!/usr/bin/env python

import v20
import os

def main():
    account_id = os.getenv('ACCOUNT_ID')

    api = v20.Context(
        hostname='stream-fxpractice.oanda.com',
        port=443,
        ssl=True,
        application=os.getenv('APPLICATION_NAME'),
        token=os.getenv('TOKEN'),
        datetime_format="UNIX"
    )

    response = api.pricing.stream(
        account_id,
        snapshot=True,
        instruments=os.getenv('INSTRUMENTS'),
    )

    for msg_type, msg in response.parts():
        print(msg_type)

        if msg_type == "pricing.Heartbeat":
            print("heartbeat")
        elif msg_type == "pricing.ClientPrice":
            print(vars(msg))


if __name__ == "__main__":
    main()
