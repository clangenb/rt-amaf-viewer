import unittest
from unittest.mock import Mock

from visualizer.matrix.leds.tcp_strips import LazyTcpStrip


class LazyLedStripShould(unittest.TestCase):

    def test_should_send_3_updated_pixels(self):
        protocol = Mock()
        protocol.sendLine()
        strip = LazyTcpStrip(protocol)

        strip.setPixelColor(1, 0x1)
        strip.setPixelColor(2, 0x2)
        strip.setPixelColor(3, 0x3)

        strip.show()

        protocol.sendLine.assert_called_with("#1,1#2,2#3,3#".encode("ascii"))


if __name__ == '__main__':
    unittest.main()
