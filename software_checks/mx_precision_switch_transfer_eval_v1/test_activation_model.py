import unittest

from activation_model import activation_traffic


class ActivationTrafficTests(unittest.TestCase):
    def test_real_input_and_unquantized_last_boundary(self):
        result=activation_traffic(1)
        self.assertEqual(result['boundaries'][0]['shape'],[1,3,40,40])
        self.assertEqual(result['boundaries'][0]['useful_bytes'],19200)
        self.assertEqual(result['boundaries'][-2]['activation_bits'],32)
        self.assertEqual(result['boundaries'][-2]['useful_bytes'],512)
        self.assertEqual(result['boundaries'][-1]['useful_bytes'],40)
    def test_independent_logical_totals(self):
        expected={32:134696,8:48872,4:34568,2:27416,1:23840}
        for bits,total in expected.items():
            with self.subTest(bits=bits):
                result=activation_traffic(bits)
                self.assertEqual(result['useful_bytes'],total)
                self.assertEqual(result['useful_bytes'],result['useful_read_bytes']+result['useful_write_bytes'])
    def test_full_line_rounding(self):
        self.assertEqual(activation_traffic(32,64)['transfer_bytes'],134720)
        self.assertEqual(activation_traffic(8,64)['transfer_bytes'],49216)
        self.assertEqual(activation_traffic(1,64)['transfer_bytes'],24384)
        for item in activation_traffic(1,64)['boundaries']:
            self.assertEqual(item['address']%64,0)
            self.assertGreaterEqual(item['transfer_bytes'],item['useful_bytes'])
            self.assertLess(item['transfer_bytes']-item['useful_bytes'],64)
    def test_batch_and_nonlinear_rounding(self):
        one,two=activation_traffic(1,64,1),activation_traffic(1,64,2)
        self.assertEqual(two['useful_bytes'],2*one['useful_bytes'])
        self.assertLess(two['transfer_bytes'],2*one['transfer_bytes'])
        self.assertEqual(two['transactions']*64,two['transfer_bytes'])
    def test_invalid_inputs(self):
        for args in ((6,64,1),(True,64,1),(8,0,1),(8,64,0),(8,64,1.5)):
            with self.subTest(args=args),self.assertRaises(ValueError): activation_traffic(*args)


if __name__=='__main__': unittest.main()
