DNN CLI
=======

A deep learning command line tool for training deep neural network AI models and running an inference server to query trained models.


RUNNING TRAINING
================

Once packaged into a JAR, training can be run from the command line:

java -cp dnn-cli.jar -train \<training-config\> \<temp-directory\> \<model-save-file\> \<eval-save-file\>

For example:

java -cp dnn-cli.jar -train training.json temp/ models/subject_classifier.model evas/subject_classifier_eval.json

**Arguments**:
  - training-config: A JSON file with the training configuration. See below for JSON config structure.
  - temp-directory: A directory into which the training session can write temporary files.
  - model-save-file: The file into which the resulting model to be saved.
  - eval-save-file: The file into which the final evaluation statistics will be saved.


JSON Training Config
--------------------

The JSON training configuration file has the following structure:

```
{
	"reuse-cached-sequence-data": false,
	"hyper-parameters": {
		"random-seed": 1249641284,
		"evaluation-metric": "f1",
		"max-epochs": 100,
		"batch-size": 128
	},
	"model-instance-builder": "au.gov.nla.dnn.model.mln.MultiLayerNetworkModelInstanceBuilder",
	"model-instance-builder-config": {
		"weight-init": "RELU",
		"activation-function": "RELU",
		"optimization-algorithm": "STOCHASTIC_GRADIENT_DESCENT",
		"updater": "nesterovs",
		"updater-params": {
			"learning-rate": 0.1,
			"momentum": 0.9
		},
		"hidden-layers": [
			{
				"node-count": 100,
				"activation-function": "RELU",
				"layer-type": "dense"
			},
			{
				"node-count": 100,
				"activation-function": "RELU",
				"layer-type": "dense"
			},
			{
				"node-count": 100,
				"activation-function": "RELU",
				"layer-type": "dense"
			}
		],
		"output-layer": {
			"loss-function": "XENT"
		},
		"backpropagation-type": "standard"
	},
	"labels": [
		"arts",
		"business_economy",
		"defence",
		"education",
		"environment",
		"government_law",
		"health",
		"history",
		"humanities",
		"indigenous_australians",
		"industry_technology",
		"media",
		"people_culture",
		"politics",
		"sciences",
		"society_social_issues",
		"sports_recreation",
		"tourism_travel"
	],
	"training-record-provider": "au.gov.nla.dnn.record.file.FileDataRecordProvider",
	"training-record-provider-config": {
		"directory": "training_data/"
	},
	"evaluation-record-provider": "au.gov.nla.dnn.record.file.FileDataRecordProvider",
	"evaluation-record-provider-config": {
		"directory": "evaluation_data/"
	},
	"input-sequence": "au.gov.nla.dnn.sequence.text.tfidf.TFIDFVectorSpaceInputSequence",
	"input-sequence-config": {
		"word-split-pattern": "[^\w]+",
		"word-exclusion-patterns": [
			"[0-9]+"
		],
		"word-min-characters": 3,
		"word-max-characters": 12,
		"word-min-occurrances": 3
	},
	
}
```

Input Sequence 
---------------

The specified input sequence is responsible for converting raw data (a byte array) into a feature set (a double array) to be fed into the model. The input sequence feeds data into the model during training, evaluation, and inference.

The example above uses the TFIDFVectorSpaceInputSequence which produces features based on a Term Frequency Inverse Document Frequency bag of words approach.

Additional input sequences can be created by implementing the InputSequence interface.

Note that if reuse-cached-sequence-data (in the training config) is set to true, the input sequence data is read from the temp directory rather than processed again.

Record Providers 
----------------

Record providers need to provide a byte array for each record. When using a text-based input sequence such as the TFIDFVectorSpaceInputSequence input sequence, as in the example above, the byte array should represent text.

The example above uses the FileDataRecordProvider class, which reads 1 file per record. It expects the specified directory to contain 1 sub-directory per label, with each label directory containing 1 file per record.

Additional record providers can be created by implementing the RawDataRecordProvider interface.

Hyper Parameters 
----------------

- random-seed: Optional. Specifies the random seed.
- evaluation-metric: The metric used to determine the model's accuracy score. Choices are 'f1', 'precision', and 'recall'.
- max-epochs: The maximum amount of epochs to train for. An epoch is a full exposure (with random batches) to all training data fed through the model with back propagation, and an evaluation to determine its score. A copy of the model at the best scoring epoch is kept as the result.
- batch-size: The number of records to load with each batch. A higher batch size improves training speed but also increases memory consumption. Batch size can also impact the way the model learns.

Model Configuration 
-------------------

The MultiLayerNetworkModelInstance will create a deep neural network with an input layer, the specified hidden layers, and output layer.

- weight-init: The weight initialisation algorithm. Options are: DISTRIBUTION, ZERO, ONES, SIGMOID_UNIFORM, NORMAL, LECUN_NORMAL, UNIFORM, XAVIER, XAVIER_UNIFORM, XAVIER_FAN_IN, XAVIER_LEGACY, RELU, RELU_UNIFORM, IDENTITY, LECUN_UNIFORM, VAR_SCALING_NORMAL_FAN_IN, VAR_SCALING_NORMAL_FAN_OUT, VAR_SCALING_NORMAL_FAN_AVG, VAR_SCALING_UNIFORM_FAN_IN, VAR_SCALING_UNIFORM_FAN_OUT, VAR_SCALING_UNIFORM_FAN_AVG.
- activation-function: The default activation function to use for each layer. Options are: CUBE, ELU, HARDSIGMOID, HARDTANH, IDENTITY, LEAKYRELU, RATIONALTANH, RELU, RELU6, RRELU, SIGMOID, SOFTMAX, SOFTPLUS, SOFTSIGN, TANH, RECTIFIEDTANH, SELU, SWISH, THRESHOLDEDRELU, GELU, MISH.
- optimization-algorithm: The optimization algorithm to use. Options are: LINE_GRADIENT_DESCENT, CONJUGATE_GRADIENT, LBFGS, STOCHASTIC_GRADIENT_DESCENT.
- updater and updater-params. The updater to use and associated parameters. The following updates are supported:
  - adadelta
  - adagrad: learning-rate, epsilon.
  - adam: learning-rate, beta1, beta2, epsilon.
  - adamax: learning-rate, beta1, beta2, epsilon.
  - amsgrad: learning-rate, beta1, beta2, epsilon.
  - nadam: learning-rate, beta1, beta2, epsilon.
  - nesterovs: learning-rate, momentum.
  - noop
  - rmsprop: learning-rate, decay, epsilon.
  - sgd: learning-rate.
- hidden-layers: An array of hidden layers. Each layer has the following configuration:
  - node-count: The number of nodes in this layer:
  - bias-init: Optional. Specifies the initial bias.
  - inverse-input-retain-probability: Optional. Specifies the dropout rate.
  - activation-function: Optional. The activation function to use. If not specified, network's default will be used. See 'activation-function' above for options.
  - loss-function: Only needs to be specified for specific layer types denoted with \*. Options are MSE, L1, XENT, MCXENT, SPARSE_MCXENT, SQUARED_LOSS, RECONSTRUCTION_CROSSENTROPY, NEGATIVELOGLIKELIHOOD, COSINE_PROXIMITY, HINGE, SQUARED_HINGE, KL_DIVERGENCE, MEAN_ABSOLUTE_ERROR, L2, MEAN_ABSOLUTE_PERCENTAGE_ERROR, MEAN_SQUARED_LOGARITHMIC_ERROR, POISSON, WASSERSTEIN.
  - layer-type: The type of layer. Depending on the type of layer, additional fields are required here, as specified below:
    - batch-normalisation:  gamma, beta, lock-gamma-beta.
    - centre-loss-output\*: alpha, lambda, gradient-check.
    - cnn-loss\*
    - convolution: kernel-size-array, stride-array, padding-array.
    - convolution-1d: kernel-size, stride, padding.
    - convolution-2d-depthwise: kernel-size-array, stride-array, padding-array.
    - convolution-2d-separable: kernel-size-array, stride-array, padding-array.
    - convolution-3d: kernel-size-array, stride-array, padding-array.
    - deconvolution-2d: kernel-size-array, stride-array, padding-array.
    - dense
    - dropout: dropout.
    - element-wise-multiplication
    - embedding
    - embedding-sequence
    - loss\*
    - lstm
    - ocnn-output
    - output\*
    - prelu
    - rnn-loss\*
    - rnn-loss-output\*
    - simple-rnn
- output-layer: Specifies the output layer configuration. Configuration syntax is the same as a hidden-layer.
- backpropagation-type: The type of back propagation to use. "standard" should be specified.

Additional model types can be supported by creating new model builders that implement the ModelInstance.Builder interface.

Evaluation Statistics
---------------------

When training is complete, the highest scoring model will be saved to file, and the corresponding evaluation statistics are saved to file in JSON format.

```
{
  "accuracy": 0.9,
  "accuracy-per-label": {
    "arts": 0.85,
    "business_economy": 0.95
    ...
  },
  "confusion-per-label": {
    "arts": {
      "true-positives": 9500,
      "true-negatives": 7500,
      "false-positives": 500,
      "false-negatives": 3500,
      "precision": 0.95,
      "recall": 0.73,
      "false-discovery-rate": 0.05,
      "false-positive-rate": 0.0625,
      "matthews-correlation-coefficient": 0.81
    }
    ...
  }
}
```


INFERENCE
=========

Once models have been trained and saved to file, you can use the following command to run a server that loads models into memory and provides remote APIs for model inference:

java -jar dnn-cli.jar -inference-server \<server-config\>

For example:

java -jar dnn-cli.jar -inference-server server.json

The server configuration must be a JSON file in the following format:

```
{
  "models": [
    {
      "id": "subject-classifier",
      "file": "models/subject_classifier.model"
    }
  ],
  "api-adapters": [
    {
      "class": "au.gov.nla.dnn.inference.api.http.HTTPInferenceApiAdapter",
      "properties-file": "http_adapter.properties"
    },
    {
      "class": "au.gov.nla.dnn.inference.api.socket.SocketInferenceApiAdapter",
      "properties-file": "socket_adapter.properties"
    }
  ]
}
```

HTTP Inference Adapter 
----------------------

The properties required to be specified in the properties file for the HTTPInferenceApiAdapter adapter specified by the properties-file key are:
  - host: The hostname on which the HTTP server socket will listen.
  - port: The port on which the HTTP server socket will listen
  - backlog: The request backlog for the HTTP server.
  - thread-pool: The number of threads on which to process HTTP requests. If 0, there will be no limit.
  - max-shutdown-delay-seconds: The maximum amount of time to wait (in seconds) for the HTTP server to shutdown when the process is killed.
  - cors-allowed: Whether or not to accept CORS requests (true/false).
  - cors-allowed-hosts: The value to send in the Access-Control-Allow-Origin response header.
  - cors-allowed-headers: The value to send in the Access-Control-Allow-Headers response header.

Note that the properties file is a key-value style properties file, not a JSON file.

To perform remote inference on a model via the HTTPInferenceApiAdapter, send a HTTP POST request to the following URL: http://host:port/infer, with the raw record data passed in the post body.
The "dnn-model-id" header is required and must specify the model ID of the model you want to infer from.

The HTTP response for a failed request will have the following JSON structure:

```
{
  "success": false,
  "exception": "The reason for the failure will be here."
}
```

The HTTP response for a successful request will have the following JSON structure:

```
{
  "success": true,
  "result": {
    "highest-label": "arts",
    "highest-confidence": 0.74,
    "scores": [
      "art": 0.74,
      "education": 0.5,
      "science_industry": 0.12
    ]
  }
}
```

Socket Inference Adapter 
------------------------

The properties required to be specified in the properties file for the SocketInferenceApiAdapter adapter specified by the properties-file key are:
  - host: The hostname on which the server socket will listen.
  - port: The port on which the server socket will listen
  - backlog: The request backlog for the server socket.
  - thread-pool: The number of threads on which to process socket requests. If 0, there will be no limit.
  - max-shutdown-delay: The maximum amount of time to wait (in milliseconds) for the adapter to shutdown when the process is killed.

Note that the properties file is a key-value style properties file, not a JSON file.

To perform remote inference on a model via the SocketInferenceApiAdapter, connect a TCP socket and send the model ID as a string, followed by the raw record body. The following example is written using Java:

```
...
try(Socket socket = new Socket(host, port))
{
  socket.getOutputStream().write(modelId.getBytes());
  socket.getOutputStream().write(rawDataBytes);
  socket.getOutputStream().flush();

  BufferedInputStream bis = new BufferedInputStream(socket.getInputStream());
  ByteArrayOutputStream buf = new ByteArrayOutputStream();
  for (int result = bis.read(); result != -1; result = bis.read()) {
      buf.write((byte)result);
  }
  
  String jsonResponse = buf.toString("UTF-8");
  ...
}
```
