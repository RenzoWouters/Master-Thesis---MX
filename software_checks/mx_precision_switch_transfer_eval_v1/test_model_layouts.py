import unittest
from model_layouts import MODELS, build_layout, route_self_check, pack_unsigned, unpack_unsigned


class LayoutTests(unittest.TestCase):
    def test_known_padding_and_counts(self):
        for mid, groups, padding in [('triple_scale64_b32',4080,1088), ('triple_scale64_b64',2160,8768)]:
            layout=build_layout(mid)
            self.assertEqual(sum(x['n_weights'] for x in layout['layer_rows']),129472)
            self.assertEqual(sum(x['groups'] for x in layout['layer_rows']),groups)
            self.assertEqual(sum(x['padded_slots'] for x in layout['layer_rows']),padding)

    def test_same_cold_low_payload_not_automatic_truncation_gain(self):
        for direct,trunc in [('dual_direct','dual_mantissa'),('triple_direct','triple_scale64_b32')]:
            a,b=build_layout(direct),build_layout(trunc)
            for mode in a['model']['modes']:
                self.assertEqual(a['mode_payloads'][str(mode)],b['mode_payloads'][str(mode)])
            self.assertLess(b['storage']['total_bytes'],a['storage']['total_bytes'])

    def test_full_source_has_no_selective_read_saving(self):
        a,b=build_layout('triple_scale64_b32','selective'),build_layout('triple_scale64_b32','full_source')
        self.assertEqual(a['storage'],b['storage'])
        self.assertEqual(b['mode_payloads']['4']['codes'],130560)
        self.assertEqual(a['mode_payloads']['4']['codes'],65280)

    def test_github_shared_scale_and_bn(self):
        a=build_layout('github5')
        self.assertEqual(a['storage']['codes'],129472*47//8)
        self.assertEqual(a['storage']['scales'],24)
        self.assertEqual(a['storage']['aux'],8808+5*2048)
        self.assertEqual(a['mode_payloads']['32']['scales'],0)
        self.assertEqual(a['mode_payloads']['1']['scales'],24)

    def test_all_mode_payload_bn_and_static_layers(self):
        for m in MODELS:
            a=build_layout(m['id'])
            self.assertEqual(a['storage']['aux'],8808+len(m['modes'])*2048)
            for p in a['mode_payloads'].values(): self.assertEqual(p['aux'],10856)

    def test_universal_code_mapping_and_byte_packing(self):
        self.assertEqual(route_self_check()['finite_e4_codes'],254)
        self.assertEqual(pack_unsigned([15,0,8],4),bytes([15,8]))
        self.assertEqual(unpack_unsigned(bytes([15,8]),4,3),[15,0,8])
        with self.assertRaises(ValueError): pack_unsigned([16],4)


if __name__=='__main__': unittest.main()
