use std::f64::consts::E;
use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {

    fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();
        let mut res = Matrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }

        res
    }

    fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data: data,
        }
    }

    fn multiply(&mut self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("ERROR: tried to multiply incompatable matrices");
        }

        let mut res = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }

        res
    }

    fn sum(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("ERROR: tried to add incompatable matrices");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        res
    }

    fn subtract(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("ERROR: tried to subtract incompatable matrices");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }

    fn dot(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("ERROR: tried to dot incompatable matrices");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        res
    }

    fn map(&mut self, function: Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|value| function(value)).collect())
            .collect()
        )
    }

    fn transpose(&mut self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }
}

fn SIGMOID(value: &f64) -> f64 {
    1.0 / (1.0 + E.powf(-value))
}

fn SIGMOID_DERIVATIVE(value: &f64) -> f64 {
    value * (1.0 - value)
}

struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
}

impl Network {

    fn new(layers: Vec<usize>, learning_rate: f64) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate: learning_rate,
        }
    }

    fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("ERROR: Trying to feed data that is incompatable with input layer");
        }

        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .sum(&self.biases[i])
                .map(SIGMOID);
            self.data.push(current.clone());
        }

        current.data[0].to_owned()
    }

    fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("ERROR: Invalid number of targets");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = errors.map(SIGMOID_DERIVATIVE);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i]
                .transpose()
                .sum(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].sum(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(SIGMOID_DERIVATIVE);
        }
    }

    fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        for i in 1..=epochs {
            if epochs > 100 || i % (epochs / 100) == 0 {
                println!("Epoch: {} of {}", i, epochs);
            }

            for j in 0..inputs.len() {
                let outs = self.feed_forward(inputs[j].clone());
                self.back_propagate(outs, targets[j].clone());
            }
        }
    }

}

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    let mut network = Network::new(vec![2, 2, 1], 0.5);
    network.train(inputs, outputs, 10000);

    println!("0, 0: {}", network.feed_forward(vec![0.0, 0.0]));
    println!("0, 1: {}", network.feed_forward(vec![0.0, 1.0]));
    println!("1, 0: {}", network.feed_forward(vec![1.0, 0.0]));
    println!("1, 1: {}", network.feed_forward(vec![1.0, 1.0]));
}
