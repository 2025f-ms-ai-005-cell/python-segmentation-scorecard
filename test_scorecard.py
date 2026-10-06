import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from scorecard import compare, evaluate, graphic, main

class Tests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(compare([[1,0]], [[1,0]])['metrics']['iou'], 1)
    def test_error_counts(self):
        r = compare([[1,1,0,0]], [[1,0,1,0]])
        self.assertEqual(r['counts'], dict(tp=1,fn=1,fp=1,tn=1))
        self.assertAlmostEqual(r['metrics']['iou'], 1/3)
        self.assertEqual(r['metrics']['dice'], .5)
    def test_empty(self):
        m = compare([[0]], [[0]])['metrics']
        self.assertIsNone(m['iou'])
        self.assertIsNone(m['precision'])
        self.assertEqual(m['accuracy'], 1)
    def test_missed(self):
        self.assertEqual(compare([[1]], [[0]])['metrics']['recall'], 0)
    def test_false_positive(self):
        self.assertEqual(compare([[0]], [[1]])['metrics']['iou'], 0)
    def test_shape(self):
        with self.assertRaises(ValueError): compare([[1]], [[1,0]])
    def test_invalid(self):
        for m in ([],[[]],[[True]],[[2]],[[0],[1,0]]):
            with self.subTest(m=m), self.assertRaises(ValueError): compare(m,m)
    def test_macro_micro(self):
        r = evaluate({'cases':[dict(name='a',reference=[[1]],prediction=[[1]]),dict(name='b',reference=[[1,1,1]],prediction=[[0,0,0]])]})
        self.assertEqual(r['micro']['iou'], .25)
        self.assertEqual(r['macro']['iou']['value'], .5)
    def test_macro_defined_count(self):
        r=evaluate({'cases':[dict(name='empty',reference=[[0]],prediction=[[0]])]})
        self.assertEqual(r['macro']['iou'], dict(value=None,defined_cases=0))
    def test_names(self):
        c=dict(name='same',reference=[[1]],prediction=[[1]])
        with self.assertRaises(ValueError): evaluate({'cases':[c,c]})
    def test_svg_escape(self):
        r=dict(name='<script>&', **compare([[1]],[[1]]))
        ET.fromstring(graphic(r))
        self.assertIn('&lt;script&gt;', graphic(r))
    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inp=p/'data.json';out=p/'out'
            inp.write_text(json.dumps({'cases':[dict(name='x',reference=[[1]],prediction=[[1]])]}))
            main([str(inp),'--out',str(out)])
            self.assertTrue((out/'index.html').exists())
            self.assertEqual(json.loads((out/'report.json').read_text())['micro']['iou'],1)
            with self.assertRaises(SystemExit): main([str(inp),'--out',str(out)])

if __name__ == '__main__': unittest.main()
