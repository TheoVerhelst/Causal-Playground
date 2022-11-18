import unittest
from causality import CausalGraph

class TestGraphHelpers(unittest.TestCase):
    def setUp(self):
        self.g = CausalGraph()
        self.g.add_node("X", (0, 0))
        self.g.add_node("Y", (5, 0))
        self.g.add_node("Z", (3, 0))
        self.g.add_node("W", (0, 3))
        self.g.add_edge("X", "Z")
        self.g.add_edge("W", "Y")

    def test_is_d_separated(self):
        # Current graph: X -> Z; W -> Y
        for a, other_a in [("Z", "X"), ("X", "Z")]:
            for b, other_b in [("W", "Y"), ("Y", "W")]:
                self.assertFalse(self.g.is_d_separated(a, other_a, set()))
                self.assertFalse(self.g.is_d_separated(b, other_b, set()))
                self.assertTrue(self.g.is_d_separated(a, b, set()))
                self.assertTrue(self.g.is_d_separated(a, b, {other_b}))
                self.assertTrue(self.g.is_d_separated(a, b, {other_a}))
                self.assertTrue(self.g.is_d_separated(a, b, {other_b, other_a}))

        self.g.add_edge("W", "X")
        # Current graph: Y <- W -> X -> Z
        self.assertFalse(self.g.is_d_separated({"X"}, {"Y"}, set()))
        self.assertFalse(self.g.is_d_separated({"Z"}, {"Y"}, set()))
        self.assertTrue(self.g.is_d_separated({"Z"}, {"W"}, {"X"}))
        self.assertTrue(self.g.is_d_separated({"Z"}, {"Y"}, {"W"}))
        self.assertTrue(self.g.is_d_separated({"Z"}, {"Y"}, {"X"}))
        self.assertTrue(self.g.is_d_separated({"X"}, {"Y"}, {"W"}))

        self.g.add_edge("Z", "Y")
        #                Y <- Z
        #                ^    ^
        #                |    |
        # Current graph: W -> X
        self.assertTrue(self.g.is_d_separated({"X"}, {"Y"}, {"Z", "W"}))
        self.assertTrue(self.g.is_d_separated({"Z"}, {"W"}, {"X"}))
        self.assertFalse(self.g.is_d_separated({"X"}, {"Y"}, set()))
        self.assertFalse(self.g.is_d_separated({"Z"}, {"W"}, {"X", "Y"}))
        self.assertFalse(self.g.is_d_separated({"Y"}, {"X"}, {"W"}))
        self.assertFalse(self.g.is_d_separated({"Y"}, {"X"}, {"Z"}))
        self.assertFalse(self.g.is_d_separated({"W"}, {"Y"}, {"Z"}))


if __name__ == '__main__':
    unittest.main()
