# Simple Linear Regression with Rust Burn Library

## Overview
This project implements a simple linear regression model using the Rust programming language and the Burn library. The model predicts the output of the function `y = 2x + 1` using synthetic data.

## Requirements

### Software Requirements
- **Rust**: Install from [Rust official site](https://www.rust-lang.org/tools/install)
- **Rust Rover IDE**: Download from [JetBrains](https://www.jetbrains.com/rust/)
- **Git**: Install from [Git](https://git-scm.com/)

### Rust Dependencies
Ensure the following dependencies are included in `Cargo.toml`:
```toml
[dependencies]
burn = { version = "0.16.0", features = ["wgpu", "train"] }
burn-ndarray = "0.16.0"
and = "0.9.0"
rgb = "0.8.50"
textplots = "0.8.6"
```

## Project Setup
1. Clone this repository:
   ```sh
   git clone <your-repo-url>
   cd linear_regression_model
   ```
2. Install Rust and dependencies:
   ```sh
   cargo build
   ```
3. Run the model:
   ```sh
   cargo run
   ```

## Model Implementation
- Generates synthetic data where `y = 2x + 1` with some noise.
- Defines a simple linear regression model using the Burn library.
- Uses Mean Squared Error as the loss function.
- Trains the model using the Adam optimizer.
- Evaluates performance and visualizes results using `textplots`.

### Output Example
The training process will display loss updates at every 100 epochs. After training, a plot of predicted vs. actual values will be displayed in the terminal.

## Challenges Faced

### 1. **Error: E0308 - Type Mismatch**
This error occurs when there is a mismatch in the expected data type of an argument passed to a function. For example, passing a string where an integer is expected can cause this error.
- **Solution**: Ensure that the arguments passed match the expected types. In my case, this happened when I mistakenly passed a string instead of an integer to the `plus_one` function.

### 2. **Error: E0599 - Method Not Found**
The error appears when trying to call a method on a type that doesn’t implement it. I encountered this while trying to call a method (`chocolate`) on a struct (`Mouth`), which wasn't defined.
- **Solution**: I had to define the method within the struct’s `impl` block to fix the error.

### 3. **Error: E0609 - Nonexistent Struct Field**
This error occurred when trying to access a field (`foo`) that didn't exist in the struct (`StructWithFields`).
- **Solution**: I checked the struct definition and used the correct field name (`x`) to resolve the issue.

### 4. **Compilation Errors with `cargo build` or `cargo run`**
- **Possible Causes**: Issues like incorrect dependencies, misused methods, or tensor shape mismatches might cause compilation errors.
- **Solution**: After encountering errors with `cargo build` or `cargo run`, I examined the error messages, ensured that the correct dependencies were included in `Cargo.toml`, and confirmed the tensor shapes were properly matched in the model.

### 5. **Dependency and API Issues**
The `burn` and related crates often undergo updates, leading to API changes. Some methods like `step()` or `forward()` were called incorrectly due to such updates.
- **Solution**: I made sure to check the documentation for any changes to the API and ensure proper usage of methods.

## Lessons Learned
- How to use the Burn library for machine learning in Rust.
- Implementing a basic neural network model for regression.
- Understanding gradient descent and loss minimization in Rust.

# Reflection on Learning

-  Sources of Help: AI, official documentation, and online community resources were instrumental in troubleshooting errors.

- Challenges: Some errors took longer to debug, especially those related to API changes.

- Takeaways: This project strengthened my understanding of Rust’s type system, error handling, and neural network implementation using the Burn library. Future improvements may involve seeking additional help from experienced Rust developers or the Burn library community.



## Author
Nelisiwe
