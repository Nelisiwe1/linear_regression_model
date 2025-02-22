use burn::backend::NdArray; // Using NdArray backend
use burn::tensor::{Tensor, Data};
use burn::optim::{Adam, Optimizer};
use burn::module::Module;
use burn::nn::Linear;
use rand::Rng;
use textplots::{Chart, Plot, Shape};

// Define the Linear Regression model
#[derive(Module, Debug, Clone)] // Derive Clone here
struct LinearRegression {
    linear: Linear<NdArray<f32>>, // Use NdArray<f32> for the backend
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            linear: Linear::new(1, 1),
        }
    }

    fn forward(&self, x: Tensor<NdArray<f32>, 2>) -> Tensor<NdArray<f32>, 2> {
        self.linear.forward(x)
    }
}

fn main() {
    // Generate synthetic data
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..100).map(|x| x as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0 + rng.gen_range(-5.0..5.0)).collect();

    // Convert to tensors
    let x_tensor = Tensor::<NdArray<f32>, 2>::from_data(Data::from(x_data)).reshape([100, 1]);
    let y_tensor = Tensor::<NdArray<f32>, 2>::from_data(Data::from(y_data)).reshape([100, 1]);

    // Initialize model and optimizer
    let mut model = LinearRegression::new();
    let mut optimizer = Adam::new(&model, 0.01);

    // Training loop
    for epoch in 0..1000 {
        // Forward pass
        let y_pred = model.forward(x_tensor.clone());  // Use clone to pass correct tensor reference

        // Compute loss (Mean Squared Error)
        let loss = (y_pred - y_tensor.clone()).powf(2.0).mean();

        // Backpropagation
        optimizer.zero_grad(); // Zero gradients before the backward pass
        loss.backward(); // Compute gradients
        optimizer.step(); // Update model parameters

        // Display loss every 100 epochs
        if epoch % 100 == 0 {
            let loss_value = loss.to_data().values[0];
            println!("Epoch {}: Loss = {}", epoch, loss_value);
        }
    }

    // Evaluation
    let y_pred = model.forward(x_tensor.clone());
    let y_pred_values: Vec<(f32, f32)> = x_data.iter().zip(y_pred.to_data().values.iter()).map(|(&x, &y)| (x, *y)).collect();

    println!("\nFinal Model Predictions:");
    Chart::new(120, 40, 0.0, 100.0)
        .lineplot(&Shape::Lines(&y_pred_values))
        .display();
}
