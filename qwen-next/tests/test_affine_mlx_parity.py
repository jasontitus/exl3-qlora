"""CPU only: qualify fixtures without importing MLX or NumPy."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from affine_mlx_parity import fixture, AffineReference, require_shape, matrix_comparison, single_output


class Fixtures(unittest.TestCase):
    def test_independent_direct_codes_and_metadata(self):
        for bits in (2,3,4,5,6,8):
            for group in (32,64,128):
                p,s,z,x,dy = fixture(bits,group,3)
                ref = AffineReference(p,s,z,in_features=2*group,bits=bits,group_size=group)
                w = ref.decode_rows(count=32)
                for o,row in enumerate(w):
                    for i,value in enumerate(row):
                        q = (o*13+i*7+i//3) % (2**bits)
                        self.assertEqual(value,s[o][i//group]*q+z[o][i//group])
                self.assertEqual((len(x),len(x[0]),len(dy[0])),(3,2*group,32))

    def test_reject_unbounded_geometry(self):
        for args in ((1,32,1),(4,256,1),(4,32,10000)):
            with self.assertRaises(ValueError):fixture(*args)

    def test_cpu_import_does_not_import_mlx(self):
        self.assertNotIn('mlx.core',sys.modules)

    def test_broadcastable_or_dropped_batch_is_rejected(self):
        for actual, expected in [((32,),(1,32)),((1,32),(3,32)),((3,32),(3,64))]:
            with self.assertRaises(ValueError):require_shape(actual,expected,'forward')
        require_shape((1,32),(1,32),'forward')

    def test_small_zero_or_reversed_gradients_fail(self):
        for dtype in ('float32','float16','bfloat16'):
            expected = [[0.0001,-0.0002]]
            self.assertTrue(matrix_comparison(expected,expected,dtype)['passed'])
            self.assertFalse(matrix_comparison([[0,0]],expected,dtype)['passed'])
            self.assertFalse(matrix_comparison([[-0.0001,0.0002]],expected,dtype)['passed'])

    def test_vjp_container_never_iterates_array_batch(self):
        class Array:
            def __iter__(self):raise AssertionError('Do not iterate batch')
        array = Array()
        for value in (array,[array],(array,)):
            self.assertIs(single_output(value),array)
        for value in ([],[array,array]):
            with self.assertRaises(ValueError):single_output(value)


if __name__ == '__main__':unittest.main()
