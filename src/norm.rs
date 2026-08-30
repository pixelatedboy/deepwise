pub struct Normalizer {
    min_vals: Vec<f64>,
    max_vals: Vec<f64>,
    target_min: Option<f64>,
    target_max: Option<f64>,
}

impl Normalizer {
    pub fn new() -> Self {
        Normalizer {
            min_vals: Vec::new(),
            max_vals: Vec::new(),
            target_min: None,
            target_max: None,
        }
    }

    pub fn fit(&mut self, X: &[Vec<f64>], y: Option<&[f64]>) {
        let n_features = X[0].len();
        self.min_vals = vec![f64::INFINITY; n_features];
        self.max_vals = vec![f64::NEG_INFINITY; n_features];
        for row in X {
            for (i, &val) in row.iter().enumerate() {
                if val < self.min_vals[i] { self.min_vals[i] = val; }
                if val > self.max_vals[i] { self.max_vals[i] = val; }
            }
        }
        if let Some(y_data) = y {
            self.target_min = Some(y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
            self.target_max = Some(y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
    }

    pub fn transform_x(&self, X: &[Vec<f64>]) -> Vec<Vec<f64>> {
        X.iter().map(|row| {
            row.iter().enumerate().map(|(i, &val)| {
                let denom = self.max_vals[i] - self.min_vals[i];
                if denom != 0.0 { (val - self.min_vals[i]) / denom } else { 0.0 }
            }).collect()
        }).collect()
    }

    pub fn transform_y(&self, y: &[f64]) -> Vec<f64> {
        if self.target_min.is_none() || self.target_max.is_none() {
            return y.to_vec();
        }
        let min = self.target_min.unwrap();
        let max = self.target_max.unwrap();
        y.iter().map(|&val| {
            let denom = max - min;
            if denom != 0.0 { (val - min) / denom } else { 0.0 }
        }).collect()
    }

    pub fn inverse_transform_y(&self, y_norm: &[f64]) -> Vec<f64> {
        if self.target_min.is_none() || self.target_max.is_none() {
            return y_norm.to_vec();
        }
        let min = self.target_min.unwrap();
        let max = self.target_max.unwrap();
        y_norm.iter().map(|&val| val * (max - min) + min).collect()
    }
}

pub struct SimpleNormalizer {
    max_vals: Vec<f64>,
}
impl SimpleNormalizer {
    pub fn new() -> Self { SimpleNormalizer { max_vals: Vec::new() } }
    pub fn fit(&mut self, X: &[Vec<f64>]) {
        let n = X[0].len();
        self.max_vals = vec![0.0; n];
        for row in X {
            for (i, &val) in row.iter().enumerate() {
                if val.abs() > self.max_vals[i] { self.max_vals[i] = val.abs(); }
            }
        }
        for v in &mut self.max_vals { if *v == 0.0 { *v = 1.0; } }
    }
    pub fn transform_x(&self, X: &[Vec<f64>]) -> Vec<Vec<f64>> {
        X.iter().map(|row| row.iter().zip(&self.max_vals).map(|(&x, &m)| x / m).collect()).collect()
    }
}
