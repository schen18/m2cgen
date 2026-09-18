from m2cgen import exporters


def test_export_to_java(trained_model):
    generated_code = exporters.export_to_java(trained_model).strip()
    assert generated_code.startswith("""
public class Model {
    public static double score(double[] input) {
        return
""".strip())


def test_export_to_python(trained_model):
    generated_code = exporters.export_to_python(trained_model).strip()
    assert generated_code.startswith("""
def score(input):
    return
""".strip())


def test_export_to_javascript(trained_model):
    generated_code = exporters.export_to_javascript(trained_model).strip()
    assert generated_code.startswith("""
function score(input) {
    return
""".strip())


def test_export_to_visual_basic(trained_model):
    generated_code = exporters.export_to_visual_basic(trained_model).strip()
    assert generated_code.startswith("""
Module Model
Function Score(ByRef inputVector() As Double) As Double
    Score =
""".strip())


def test_export_to_c_sharp(trained_model):
    generated_code = exporters.export_to_c_sharp(trained_model).strip()
    assert generated_code.startswith("""
namespace ML {
    public static class Model {
        public static double Score(double[] input) {
            return
""".strip())


def test_export_to_powershell(trained_model):
    generated_code = exporters.export_to_powershell(trained_model).strip()
    assert generated_code.startswith("""
function Score([double[]] $InputVector) {
    return
""".strip())


def test_export_to_php(trained_model):
    generated_code = exporters.export_to_php(trained_model).strip()
    assert generated_code.startswith("""
<?php
function score(array $input) {
    return
""".strip())


def test_export_to_rust(trained_model):
    generated_code = exporters.export_to_rust(trained_model).strip()
    assert generated_code.startswith("""
fn score(input: Vec<f64>) -> f64 {
""".strip())
