"""Independent, exhaustive stdlib tests of the finite mantissa mappings.

The oracle decoder is an explicit, independently tabulated codebook in eighths;
it does not call or reuse the production decoding formula.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mantissa_rules import METHODS, LABELS, decode_code, map_code, lookup, numeric_rows, numeric_summary


SOURCE_EIGHTHS = (
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,
)
TARGET_EIGHTHS = (0,4,8,12,16,24,32,48)
MSB_POSITIVE_REFERENCE = (
    0,1,0,1,0,1,2,1,2,3,2,3,2,3,4,3,
    4,5,4,5,4,5,6,5,6,7,6,7,6,7,6,7,
)


def oracle_decode(code, bits):
    if bits == 6:
        magnitude = Fraction(SOURCE_EIGHTHS[code % 32], 8)
        return -magnitude if code >= 32 else magnitude
    if bits == 4:
        magnitude = Fraction(TARGET_EIGHTHS[code % 8], 8)
        return -magnitude if code >= 8 else magnitude
    raise AssertionError('Unsupported test codebook')


class CodebookTests(unittest.TestCase):
    def test_exact_decoder_matches_independent_tables(self):
        for bits in (4,6):
            for code in range(1 << bits):
                with self.subTest(bits=bits,code=code):
                    self.assertIsInstance(decode_code(code,bits),Fraction)
                    self.assertEqual(decode_code(code,bits),oracle_decode(code,bits))

    def test_format_boundaries(self):
        self.assertEqual(decode_code(1,6),Fraction(1,8))
        self.assertEqual(decode_code(7,6),Fraction(7,8))
        self.assertEqual(decode_code(8,6),Fraction(1))
        self.assertEqual(decode_code(31,6),Fraction(15,2))
        self.assertEqual(decode_code(1,4),Fraction(1,2))
        self.assertEqual(decode_code(2,4),Fraction(1))
        self.assertEqual(decode_code(7,4),Fraction(6))
        self.assertEqual(decode_code(63,6),Fraction(-15,2))
        self.assertEqual(decode_code(15,4),Fraction(-6))

    def test_validation(self):
        for bits in (0,2,5,8,True,4.0,'4'):
            with self.subTest(bits=bits),self.assertRaises(ValueError):
                decode_code(0,bits)
        for bits in (4,6):
            for code in (-1,1 << bits,True,1.0,'1'):
                with self.subTest(bits=bits,code=code),self.assertRaises(ValueError):
                    decode_code(code,bits)
        for method in ('', 'unknown', None):
            with self.subTest(method=method),self.assertRaises(ValueError):
                map_code(0,method)
            with self.subTest(method=method),self.assertRaises(ValueError):
                lookup(method)


class MappingTests(unittest.TestCase):
    def test_lookup_domain_sign_symmetry_and_signed_zeros(self):
        self.assertEqual(METHODS,('lsb_trunc','rne','msb_bestexp'))
        self.assertEqual(set(LABELS),set(METHODS))
        for method in METHODS:
            table=lookup(method)
            self.assertIsInstance(table,tuple)
            self.assertEqual(len(table),64)
            self.assertEqual(table,tuple(map_code(code,method) for code in range(64)))
            for code,target in enumerate(table):
                self.assertIs(type(target),int)
                self.assertTrue(0 <= target < 16)
                self.assertEqual(target >> 3,code >> 5)
            for code in range(32):
                self.assertEqual(table[code+32],table[code]+8)
            self.assertEqual(table[0],0)
            self.assertEqual(table[32],8)
            self.assertEqual(oracle_decode(table[0],4),0)
            self.assertEqual(oracle_decode(table[32],4),0)

    def test_lsb_is_exact_bit_shift(self):
        for code in range(64):
            self.assertEqual(map_code(code,'lsb_trunc'),code >> 2)

    def test_rne_is_globally_nearest_with_even_ties(self):
        tie_count=0
        for code in range(64):
            source=oracle_decode(code,6)
            target=map_code(code,'rne')
            chosen_error=abs(oracle_decode(target,4)-source)
            all_errors=[abs(oracle_decode(candidate,4)-source) for candidate in range(16)]
            self.assertEqual(chosen_error,min(all_errors),code)
            same_sign=[candidate for candidate in range(16) if candidate >> 3 == code >> 5]
            closest=[candidate for candidate in same_sign
                     if abs(oracle_decode(candidate,4)-source)==chosen_error]
            if len(closest)>1:
                tie_count+=1
                self.assertEqual(target & 1,0,code)
        self.assertGreater(tie_count,0)

    def test_rne_concrete_ties_and_maximum(self):
        for source,target in ((2,0),(6,2),(10,2),(14,4),(18,4),(22,6),(26,6),(31,7)):
            self.assertEqual(map_code(source,'rne'),target,(source,target))
            self.assertEqual(map_code(source+32,'rne'),target+8,(source,target))

    def test_msb_all_exponents_and_stated_tiebreak(self):
        ties=0
        for code in range(64):
            source=oracle_decode(code,6)
            target=map_code(code,'msb_bestexp')
            source_exponent=(code // 8) % 4
            self.assertEqual(target % 2,code % 2)
            self.assertEqual(target // 8,code // 32)
            candidates=[candidate for candidate in range(16)
                        if candidate // 8 == code // 32 and candidate % 2 == code % 2]
            self.assertEqual(len(candidates),4)
            candidate_errors={candidate:abs(oracle_decode(candidate,4)-source)
                              for candidate in candidates}
            optimum=min(candidate_errors.values())
            self.assertEqual(candidate_errors[target],optimum,code)
            closest=[candidate for candidate in candidates if candidate_errors[candidate]==optimum]
            if len(closest)>1:
                ties+=1
                minimum_movement=min(abs((candidate // 2)%4-source_exponent) for candidate in closest)
                least_moved=[candidate for candidate in closest
                             if abs((candidate // 2)%4-source_exponent)==minimum_movement]
                self.assertEqual((target // 2)%4,min((candidate // 2)%4 for candidate in least_moved))
        self.assertGreater(ties,0)

    def test_msb_independent_complete_reference_and_boundaries(self):
        self.assertEqual(lookup('msb_bestexp')[:32],MSB_POSITIVE_REFERENCE)
        for source,target in ((1,1),(4,0),(6,2),(7,1),(12,2),(14,4),
                              (17,5),(20,4),(22,6),(25,7),(30,6),(31,7)):
            self.assertEqual(map_code(source,'msb_bestexp'),target,(source,target))


class ReportingTests(unittest.TestCase):
    def test_rows_are_complete_consistent_and_json_serializable(self):
        rows=numeric_rows()
        self.assertEqual(len(rows),192)
        self.assertEqual(len({(r['source_code'],r['method']) for r in rows}),192)
        self.assertEqual([(r['source_code'],r['method']) for r in rows],
                         [(code,method) for code in range(64) for method in METHODS])
        self.assertEqual(json.loads(json.dumps(rows,allow_nan=False)),rows)
        for row in rows:
            source=oracle_decode(row['source_code'],6)
            target=oracle_decode(row['target_code'],4)
            error=target-source
            self.assertEqual(row['source_value'],float(source))
            self.assertEqual(row['target_value'],float(target))
            self.assertEqual(row['error'],float(error))
            self.assertEqual(row['abs_error'],float(abs(error)))
            self.assertEqual(row['squared_error'],float(error*error))
            self.assertEqual(row['magnitude_error'],float(abs(target)-abs(source)))
            self.assertEqual(row['exact'],error==0)
            if source==0:
                self.assertIsNone(row['relative_abs_error'])
            else:
                self.assertEqual(row['relative_abs_error'],float(abs(error)/abs(source)))

    def test_population_counts_and_aggregates_recomputed_from_oracle(self):
        report=numeric_summary()
        self.assertEqual(json.loads(json.dumps(report,allow_nan=False)),report)
        groups={'all64':range(64),'positive32':range(32),
                'positive_nonzero31':range(1,32),'positive_normal24':range(8,32)}
        for name,codes in groups.items():
            population=report['populations'][name]
            self.assertEqual(population['count'],len(codes))
            for method in METHODS:
                metrics=population['methods'][method]
                sources=[oracle_decode(code,6) for code in codes]
                targets=[oracle_decode(map_code(code,method),4) for code in codes]
                errors=[abs(t-s) for s,t in zip(sources,targets)]
                baseline=[abs(oracle_decode(code >> 2,4)-oracle_decode(code,6)) for code in codes]
                relative=[error/abs(source) for source,error in zip(sources,errors) if source]
                expected={'mae':sum(errors,Fraction(0))/len(codes),
                          'mse':sum((e*e for e in errors),Fraction(0))/len(codes),
                          'max_abs_error':max(errors),
                          'magnitude_bias':sum((abs(t)-abs(s) for s,t in zip(sources,targets)),Fraction(0))/len(codes),
                          'mean_relative_abs_error':sum(relative,Fraction(0))/len(relative)}
                for key,value in expected.items():
                    self.assertEqual(Fraction(metrics['exact_fractions'][key]),value,(name,method,key))
                    self.assertEqual(metrics[key],float(value),(name,method,key))
                self.assertEqual(metrics['relative_error_count'],len(relative))
                self.assertEqual(metrics['exact_count'],sum(e==0 for e in errors))
                self.assertEqual(metrics['better_than_lsb'],sum(e<b for e,b in zip(errors,baseline)))
                self.assertEqual(metrics['worse_than_lsb'],sum(e>b for e,b in zip(errors,baseline)))
                self.assertEqual(metrics['equal_to_lsb'],sum(e==b for e,b in zip(errors,baseline)))
                self.assertEqual(sum(metrics[k] for k in ('better_than_lsb','worse_than_lsb','equal_to_lsb')),len(codes))

    def test_known_uniform_metrics_and_monotonicity(self):
        report=numeric_summary()
        all64=report['populations']['all64']['methods']
        for method,mae,mse in (('lsb_trunc','3/8','77/256'),
                               ('rne','9/32','49/256'),
                               ('msb_bestexp','9/16','185/256')):
            self.assertEqual(all64[method]['exact_fractions']['mae'],mae)
            self.assertEqual(all64[method]['exact_fractions']['mse'],mse)
        msb=all64['msb_bestexp']
        self.assertEqual((msb['better_than_lsb'],msb['worse_than_lsb'],msb['equal_to_lsb']),(8,18,38))
        self.assertEqual(msb['exact_count'],8)
        self.assertEqual(msb['max_abs_error'],3)
        for method in METHODS:
            values=[oracle_decode(map_code(code,method),4) for code in range(32)]
            expected=[i for i in range(31) if values[i+1]<values[i]]
            mono=report['monotonicity'][method]
            self.assertEqual(mono['nondecreasing'],not expected)
            self.assertEqual(mono['decreases'],len(expected))
            self.assertEqual([v['source_code_before'] for v in mono['violations']],expected)
        self.assertTrue(report['monotonicity']['lsb_trunc']['nondecreasing'])
        self.assertTrue(report['monotonicity']['rne']['nondecreasing'])
        self.assertEqual(report['monotonicity']['msb_bestexp']['decreases'],12)
        checks=report['exhaustive_checks']
        self.assertEqual(checks['codes_checked'],64)
        self.assertTrue(all(value is True for key,value in checks.items() if key!='codes_checked'))
        self.assertFalse(report['assumptions']['network_accuracy_optimality_claimed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
