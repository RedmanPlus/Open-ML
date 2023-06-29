use crate::network::Network;
use crate::activation::SIGMOID;

pub mod matrix;
pub mod network;
pub mod activation;

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
    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 0.5);
    network.train(inputs, outputs, 10000);

    println!("0, 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("0, 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1, 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("1, 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}
