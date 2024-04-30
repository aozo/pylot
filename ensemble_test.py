import unittest
from pylot.perception.detection.ensemble import calculate_ensemble


class TestEnsemble(unittest.TestCase):
    def test_majority_cls(self):
        det1 = [([0.0, 0.0, 0.5, 0.5], 1.0, 1)]
        det2 = [([0.1, 0.1, 0.51, 0.51], 0.5, 1)]
        det3 = [([0.2, 0.2, 0.52, 0.52], 1.0, 2)]
        num_detections, bboxes, scores, classes = calculate_ensemble(det1, det2, det3, 1280, 720)

        self.assertEqual([[0.05, 0.05, 0.505, 0.505]], bboxes)
        self.assertEqual([0.75], scores)
        self.assertEqual([1], classes)

    def test_no_detection(self):
        det1 = [([0.0, 0.0, 0.25, 0.25], 0.5, 1)]
        det2 = [([0.0, 0.0, 0.25, 0.25], 1.0, 2)]
        det3 = [([0.0, 0.0, 0.25, 0.25], 0.75, 3)]
        num_detections, bboxes, scores, classes = calculate_ensemble(det1, det2, det3, 1280, 720)

        self.assertEqual([], bboxes)
        self.assertEqual([], scores)
        self.assertEqual([], classes)

    def test_different_num_detections(self):
        det1 = [([0.5, 0.5, 0.75, 0.75], 0.75, 3),
                ([0.0, 0.0, 0.25, 0.25], 1.0, 1)]
        det2 = [([0.0, 0.0, 0.25, 0.25], 0.5, 1)]
        det3 = [([0.25, 0.25, 0.5, 0.5], 1.0, 4),
                ([0.5, 0.5, 0.75, 0.75], 1.0, 5),
                ([0.0, 0.0, 0.25, 0.25], 1.0, 2)]
        num_detections, bboxes, scores, classes = calculate_ensemble(det1, det2, det3, 1280, 720)

        self.assertEqual([[0.0, 0.0, 0.25, 0.25]], bboxes)
        self.assertEqual([0.75], scores)
        self.assertEqual([1], classes)


if __name__ == '__main__':
    unittest.main()
