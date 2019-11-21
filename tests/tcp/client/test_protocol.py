import unittest
from unittest.mock import Mock, call

from tcp.client.protocol import LedReceiverProtocol


class LedReceiverProtocolShould(unittest.TestCase):
    def test_receive_line_correctly(self):
        led_strip = Mock()
        calls = [call.setPixelColor(1, 0x1),
                 call.setPixelColor(2, 0x2),
                 call.setPixelColor(3, 0x3)]

        protocol = LedReceiverProtocol(led_strip)
        protocol.lineReceived("#1,1#2,2#3,3#".encode("ascii"))
        print(led_strip.mock_calls)
        led_strip.assert_has_calls(calls, any_order=True)


if __name__ == '__main__':
    unittest.main()
