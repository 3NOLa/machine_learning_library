# **NetLearn-C: A From-Scratch Machine Learning Library in C**

**NetLearn-C **is a foundational machine learning and neural network library implemented entirely in C. It emphasizes clarity, modular design, and educational value—ideal for those interested in building AI systems from the ground up, without dependencies or frameworks.

This library mirrors the structure found in professional ML libraries like TensorFlow or PyTorch, but is handcrafted for deep understanding and maximum control.

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **Core Highlights**
	•	Low-level implementation of neural networks and ML components in C
	•	Custom Matrix & Tensor Libraries for numerical computation
	•	Backpropagation support for deep learning training loops
	•	Support for Classification tasks, loss functions, and activation layers
	•	Progress documented in learning_progression.txt, resembling a devlog or book chapter format

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **File Breakdown**

## **Neural Network Components**
	•	Neuron.c/h – Encapsulates neuron structure and core functions (e.g., activation, weight handling).
	•	Layer.c/h – Implements layers of neurons, connection initialization, and feedforward.
	•	Network.c/h – Coordinates the full network architecture, training steps, weight updates, and backpropagation.

## **Machine Learning Logic**
	•	Classification.c/h – Contains logic for classification models and predictions.
	•	Loss_Functions.c – Implements loss functions like Mean Squared Error and Cross-Entropy for optimization.

## **Tensor & Matrix Engine**
	•	Tensor_library.c & tensor.h – Designed for N-dimensional array manipulation similar to NumPy tensors.
	•	Matrix_library.c (from earlier versions) – May be deprecated in favor of Tensor_library.

## **Activation Functions**
	•	Activation_Functions.c & active_functions.h – Includes key activation functions like ReLU, Sigmoid, and Tanh.

## **Utilities & Tests**
	•	check.c – Contains unit tests and debugging helpers for ensuring correctness.
	•	test_library.c – Entry point for validating all library features.
	•	.vcxproj files – Visual Studio project files for easy Windows compilation and development.

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **AI Concepts Implemented**
	•	Feedforward Neural Networks
	•	Backpropagation Algorithm
	•	Gradient Descent
	•	Loss Function Minimization
	•	Tensor Operations
	•	Binary Classification

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻


## **📚 Educational Foundation**

This library was built while studying the open-access textbook:

Mathematics for Machine Learning
By Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong

It serves as the theoretical backbone of the implementation. Each algorithm in this library was carefully aligned with the concepts covered in that book.
https://mml-book.github.io/book/mml-book.pdf
⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **Build & Run**
	1.	Clone the repo:

git clone https://github.com/3NOLa/machine_learning_library.git
cd machine_learning_library/machine_learning_library


	2.	Open machine_learning_library.vcxproj in Visual Studio and build the solution.
	3.	Run test_library.c to test the core functionality:

./Debug/test_library.exe



⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **Future Roadmap (Suggestions)**
	•	Add convolution layers for image processing
	•	Expand to support regression
	•	Integrate simple GPU backend using CUDA or OpenCL (optional challenge)

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## **License**

MIT License — open for academic, personal, or hobbyist use.


