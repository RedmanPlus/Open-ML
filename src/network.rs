use crate::matrix::Matrix; 
use crate::activation::Activation;

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation<'a>,
    learning_rate: f64,
}

impl Network<'_> {

    pub fn new<'a>(
            layers: Vec<usize>,
            activation: Activation<'a>,
            learning_rate: f64
    ) -> Network<'a> {
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
            activation: activation,
            learning_rate: learning_rate,
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        println!("{:?}", inputs.len());
        println!("{:?}", self.layers[0]);
        if inputs.len() != self.layers[0] {
            panic!("ERROR: Trying to feed data that is incompatable with input layer");
        }

        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            println!("{:?}", self.weights[i]);
            println!("{:?}", self.biases[i]);
            current = self.weights[i]
                .multiply(&current)
                .sum(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }

        current.data[0].to_owned()
    }

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("ERROR: Invalid number of targets");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = errors.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i]
                .transpose()
                .sum(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].sum(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
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
