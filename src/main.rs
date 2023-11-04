use candle_core::{Device, DType, Tensor, D, Module};
use candle_nn::{Linear, VarBuilder, VarMap, ops, loss, Optimizer};
use anyhow;

const VOTE_DIM: usize = 3;
const RESULTS: usize = 1;
const EPOCHS: usize = 20;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 2;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_votes: Tensor,
    pub train_results: Tensor,
    pub test_votes: Tensor,
    pub test_results: Tensor,
}

struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> Result<Self, anyhow::Error> {
        let ln1 = candle_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, 3, vs.pp("ln3"))?;  // Changed RESULTS + 1 to 3
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, anyhow::Error> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        Ok(self.ln3.forward(&xs)?)
    }
    fn vote(&self, vote1: u32, vote2: u32, dev: &Device) -> anyhow::Result<u32> {
        let votes = vec![vote1, vote2, 0];
        let tensor_votes = Tensor::from_vec(votes.clone(), (1, VOTE_DIM), dev)?.to_dtype(DType::F32)?;
        let result_tensor = self.forward(&tensor_votes)?;
        let result = result_tensor
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .get(0).map(|x| x.to_scalar::<u32>())??;
        Ok(result)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let train_results = m.train_results.to_device(dev)?;
    let train_votes = m.train_votes.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE).unwrap();
    let test_votes = m.test_votes.to_device(dev)?;
    let test_results = m.test_results.to_device(dev)?;
    let mut final_accuracy: f32 = 0.0;
    for epoch in 1..EPOCHS+1 {
        let logits = model.forward(&train_votes)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_results)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_votes)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_results)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_results.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                 loss.to_scalar::<f32>()?,
                 final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}

pub fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let train_votes_vec: Vec<u32> = vec![
        15, 10, 0,
        10, 15, 2,
        3, 3, 1,
        5, 12, 2,
        30, 20, 0,
        16, 12, 0,
        66, 66, 1,
        13, 25, 2,
        6, 14, 2,
        31, 21, 0,
    ];
    let train_votes_tensor = Tensor::from_vec(train_votes_vec.clone(), (train_votes_vec.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let train_results_vec: Vec<u32> = vec![
        0,
        2,
        1,
        2,
        0,
        0,
        1,
        2,
        2,
        0,
    ];
    let train_results_tensor = Tensor::from_vec(train_results_vec, train_votes_vec.len() / VOTE_DIM, &dev)?;

    let test_votes_vec: Vec<u32> = vec![
        13, 9, 0,
        8, 14, 2,
        3, 10, 2,
        345, 345, 1,
        29, 14, 0,
        1423, 122, 0,
        456, 1333, 2,
        935, 1990, 2,
        22, 22, 1,
        1337, 42, 0,
    ];
    let test_votes_tensor = Tensor::from_vec(test_votes_vec.clone(), (test_votes_vec.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let test_results_vec: Vec<u32> = vec![
        0,
        2,
        2,
        1,
        0,
        0,
        2,
        2,
        1,
        0

    ];
    let test_results_tensor = Tensor::from_vec(test_results_vec.clone(), test_results_vec.len(), &dev)?;

    let m = Dataset {
        train_votes: train_votes_tensor,
        train_results: train_results_tensor,
        test_votes: test_votes_tensor,
        test_results: test_results_tensor,
    };

    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            },
            Err(e) => {
                println!("Error: {:?}", e);
                continue;
            }
        }

    }

    loop {
        println!("Enter two vote results separated by space (or 'q' to quit):");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input = input.trim().to_string();

        if input.to_lowercase() == "exit" || input.to_lowercase() == "q" {
            break;
        }

        let votes: Vec<&str> = input.split_whitespace().collect();
        if votes.len() != 2 {
            println!("Please enter exactly two vote values.");
            continue;
        }

        let vote1: u32 = match votes[0].parse() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid input for vote1. Please enter a valid number.");
                continue;
            }
        };

        let vote2: u32 = match votes[1].parse() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid input for vote2. Please enter a valid number.");
                continue;
            }
        };

        let result = trained_model.vote(vote1, vote2, &dev)?;
        println!("The voting results: [{}, {}]", vote1, vote2);
        match result {
            0 => println!("The first voting choice wins"),
            1 => println!("It's a draw!"),
            2 => println!("The seccond voting choice wins"),
            _ => println!("Unexpected result: {:?}", result),
        }
    }

    Ok(())
}
