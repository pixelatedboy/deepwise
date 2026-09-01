use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use serde_json::Value;
use serde_pyobject::{from_pyobject, to_pyobject};

#[pyfunction(name = "save")]
pub fn save(data: Bound<'_, PyAny>, path: PathBuf) -> PyResult<()> {
    let value: Value = from_pyobject(data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    let encoded = rmp_serde::to_vec(&value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    let mut file = File::create(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e))
    })?;

    file.write_all(&encoded).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write to file: {}", e))
    })?;

    Ok(())
}

#[pyfunction(name = "load")]
pub fn load(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let mut file = File::open(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
    })?;

    let value: Value = rmp_serde::from_slice(&buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    let py_obj = to_pyobject(py, &value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;

    Ok(py_obj.unbind())
}
