package au.gov.nla.dnn.model.mln;

import java.io.InputStream;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PReLULayer;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import au.gov.nla.dnn.model.ModelInstance;

public class MultiLayerNetworkModelInstanceBuilder implements ModelInstance.Builder<MultiLayerNetworkModelInstance>
{
    public MultiLayerNetworkModelInstance create(JSONObject config, int featureCount, int labelCount, long randomSeed) throws Exception
    {
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .weightInit(getWeightInit(config.getString("weight-init")))
                .activation(getActivationFunction(config.getString("activation-function")))
                .optimizationAlgo(getOptimizationAlgorithm(config.getString("optimization-algorithm")))
                .updater(getUpdater(config.getString("updater"), config.getJSONObject("updater-params")))
                .seed(randomSeed);
        
        NeuralNetConfiguration.ListBuilder definitionBuilder = b.list();
        int inputNodes = featureCount;
        int outputNodes = labelCount;
        int lastNodes = inputNodes;
        
        // Hidden layers
        
        JSONArray hiddenLayerArray = config.getJSONArray("hidden-layers");
        
        for(int i=0; i<hiddenLayerArray.length(); i++)
        {
            JSONObject layerConfig = hiddenLayerArray.getJSONObject(i);
            definitionBuilder.layer(i, createHiddenLayer(layerConfig, lastNodes, 
                    getActivationFunction(config.getString("activation-function"))));
            lastNodes = layerConfig.getInt("node-count");
        }
        
        // Add output layer
        
        definitionBuilder.layer(createOutputLayer(config.getJSONObject("output-layer"), lastNodes, outputNodes));
        
        // Specify back propagation
        
        switch(config.getString("backpropagation-type"))
        {
            case "standard":
            {
                definitionBuilder.backpropType(BackpropType.Standard);
                break;
            }
            case "truncated-bptt":
            {
                definitionBuilder.backpropType(BackpropType.TruncatedBPTT);
                
                if(config.has("bptt-length"))
                {
                    definitionBuilder.tBPTTLength(config.getInt("bptt-length"));
                }
                if(config.has("bptt-forward-length"))
                {
                    definitionBuilder.tBPTTForwardLength(config.getInt("bptt-forward-length"));
                }
                if(config.has("bptt-backward-length"))
                {
                    definitionBuilder.tBPTTBackwardLength(config.getInt("bptt-backward-length"));
                }
                break;
            }
            default:
            {
                throw new Exception("Back propagation type ["+config.getString("backpropagation-type")+"] not defined correctly.");
            }
        }
        
        // Build the network
        
        definitionBuilder.setInputType(InputType.feedForward(inputNodes));
        definitionBuilder.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
        definitionBuilder.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        
        return new MultiLayerNetworkModelInstance(new MultiLayerNetwork(definitionBuilder.build()));
    }
    
    private Layer createHiddenLayer(JSONObject config, int inputNodes, Activation defaultActivationFunction) throws Exception
    {
        return createLayer(config, inputNodes, false, config.getInt("node-count"), null, defaultActivationFunction);
    }
    
    private Layer createOutputLayer(JSONObject config, int inputNodes, int outputNodes) throws Exception
    {
        return createLayer(config, inputNodes, true, outputNodes, config.getString("loss-function"), Activation.SOFTMAX);
    }
    
    private Layer createLayer(JSONObject config, int inputNodes, boolean outputLayer, int outputNodes, String lossFunction, 
            Activation defaultActivationFunction) throws Exception
    {
        FeedForwardLayer.Builder<?> b;
        
        switch(config.getString("layer-type"))
        {
            case "batch-normalisation":
            {
                b = new BatchNormalization.Builder(
                        config.getDouble("gamma"),
                        config.getDouble("beta"),
                        config.getBoolean("lock-gamma-beta"));
                break;
            }
            case "centre-loss-output":
            {
                CenterLossOutputLayer.Builder oB = (lossFunction==null||lossFunction.isBlank())?new CenterLossOutputLayer.Builder()
                        :new CenterLossOutputLayer.Builder(LossFunction.valueOf(lossFunction));
                oB.alpha(config.getDouble("alpha"));
                oB.lambda(config.getDouble("lambda"));
                oB.gradientCheck(config.getBoolean("gradient-check"));
                b = oB;
                break;
            }
            case "cnn-loss":
            {
                b = (lossFunction==null||lossFunction.isBlank())?new CnnLossLayer.Builder()
                        :new CnnLossLayer.Builder(LossFunction.valueOf(lossFunction));
                break;
            }
            case "convolution":
            {
                b = new ConvolutionLayer.Builder(
                        parseIntParamArray(config.getJSONArray("kernel-size-array")),
                        parseIntParamArray(config.getJSONArray("stride-array")),
                        parseIntParamArray(config.getJSONArray("padding-array")));
                break;
            }
            case "convolution-1d":
            {
                b = new Convolution1DLayer.Builder(
                        config.getInt("kernel-size"), 
                        config.getInt("stride"),
                        config.getInt("padding"));
                break;
            }
            case "convolution-2d-depthwise":
            {
                b = new DepthwiseConvolution2D.Builder(
                        parseIntParamArray(config.getJSONArray("kernel-size-array")),
                        parseIntParamArray(config.getJSONArray("stride-array")),
                        parseIntParamArray(config.getJSONArray("padding-array")));
                break;
            }
            case "convolution-2d-separable":
            {
                b = new SeparableConvolution2D.Builder(
                        parseIntParamArray(config.getJSONArray("kernel-size-array")),
                        parseIntParamArray(config.getJSONArray("stride-array")),
                        parseIntParamArray(config.getJSONArray("padding-array")));
                break;
            }
            case "convolution-3d":
            {
                b = new Convolution3D.Builder(
                        parseIntParamArray(config.getJSONArray("kernel-size-array")),
                        parseIntParamArray(config.getJSONArray("stride-array")),
                        parseIntParamArray(config.getJSONArray("padding-array")),
                        parseIntParamArray(config.getJSONArray("dilation-array")));
                break;
            }
            case "deconvolution-2d":
            {
                b = new Deconvolution2D.Builder(
                        parseIntParamArray(config.getJSONArray("kernel-size-array")),
                        parseIntParamArray(config.getJSONArray("stride-array")),
                        parseIntParamArray(config.getJSONArray("padding-array")));
                break;
            }
            case "dense":
            {
                b = new DenseLayer.Builder();
                break;
            }
            case "dropout":
            {
                b = new DropoutLayer.Builder(
                        config.getDouble("dropout"));
                break;
            }
            case "element-wise-multiplication":
            {
                b = new ElementWiseMultiplicationLayer.Builder();
                break;
            }
            case "embedding":
            {
                b = new EmbeddingLayer.Builder();
                break;
            }
            case "embedding-sequence":
            {
                b = new EmbeddingSequenceLayer.Builder();
                break;
            }
            case "loss":
            {
                b = (lossFunction==null||lossFunction.isBlank())?new LossLayer.Builder():
                    new LossLayer.Builder(LossFunction.valueOf(lossFunction));
                break;
            }
            case "lstm":
            {
                b = new LSTM.Builder();
                break;
            }
            case "ocnn-output":
            {
                b = new OCNNOutputLayer.Builder();
                break;
            }
            case "output":
            {
                b = (lossFunction==null||lossFunction.isBlank())?new OutputLayer.Builder():
                    new OutputLayer.Builder(LossFunction.valueOf(lossFunction));
                break;
            }
            case "prelu":
            {
                b = new PReLULayer.Builder();
                break;
            }
            case "rnn-loss":
            {
                b = (lossFunction==null||lossFunction.isBlank())?new RnnLossLayer.Builder():
                    new RnnLossLayer.Builder(LossFunction.valueOf(lossFunction));
                break;
            }
            case "rnn-loss-output":
            {
                b = (lossFunction==null||lossFunction.isBlank())?new RnnOutputLayer.Builder():
                    new RnnOutputLayer.Builder(LossFunction.valueOf(lossFunction));
                break;
            }
            case "simple-rnn":
            {
                b = new SimpleRnn.Builder();
                break;
            }
            default:
            {
                throw new Exception("Layer type ["+config.getString("layer-type")+"] not defined correctly.");
            }
        }
        
        b.nIn(inputNodes);
        
        if(config.has("activation-function"))
        {
            b.activation(getActivationFunction(config.getString("activation-function")));
        }
        else
        {
            b.activation(defaultActivationFunction);
        }
        if(config.has("updater"))
        {
            b.updater(getUpdater(config.getString("updater"), config.getJSONObject("updater-params")));
        }
        if(config.has("bias-init"))
        {
            b.biasInit(config.getDouble("bias-init"));
        }
        if(config.has("inverse-input-retain-probability"))
        {
            b.dropOut(config.getDouble("inverse-input-retain-probability"));
        }
        
        b.nOut(outputNodes);
        return b.build();
    }
    
    private int[] parseIntParamArray(JSONArray array)
    {
        int[] r = new int[array.length()];
        
        for(int i=0; i<array.length(); i++)
        {
            r[i] = array.getInt(i);
        }
        
        return r;
    }
    
    private WeightInit getWeightInit(String value)
    {
        if(value==null || value.isEmpty())
        {
            return WeightInit.RELU;
        }
        
        return WeightInit.valueOf(value);
    }
    
    private Activation getActivationFunction(String value)
    {
        if(value==null || value.isEmpty())
        {
            return Activation.RELU;
        }
        
        return Activation.valueOf(value);
    }
    
    private OptimizationAlgorithm getOptimizationAlgorithm(String value)
    {
        if(value==null || value.isEmpty())
        {
            return OptimizationAlgorithm.LINE_GRADIENT_DESCENT;
        }
        
        return OptimizationAlgorithm.valueOf(value);
    }
    
    private IUpdater getUpdater(String value, JSONObject params) throws Exception
    {
        if(value==null || value.isEmpty())
        {
            return new AdaGrad(0.03);
        }
        
        switch(value)
        {
            case "adadelta":
            {
                return new AdaDelta();
            }
            case "adagrad":
            {
                return new AdaGrad(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "adam":
            {
                return new Adam(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("beta1")),
                        Double.parseDouble(params.getString("beta2")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "adamax":
            {
                return new AdaMax(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("beta1")),
                        Double.parseDouble(params.getString("beta2")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "amsgrad":
            {
                return new AMSGrad(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("beta1")),
                        Double.parseDouble(params.getString("beta2")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "nadam":
            {
                return new Nadam(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("beta1")),
                        Double.parseDouble(params.getString("beta2")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "nesterovs":
            {
                return new Nesterovs(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("momentum")));
            }
            case "noop":
            {
                return new NoOp();
            }
            case "rmsprop":
            {
                return new RmsProp(
                        Double.parseDouble(params.getString("learning-rate")),
                        Double.parseDouble(params.getString("rms-decay")),
                        Double.parseDouble(params.getString("epsilon")));
            }
            case "sgd":
            {
                return new Nadam(
                        Double.parseDouble(params.getString("learning-rate")));
            }
            default:
            {
                throw new Exception("Neural network updater ["+value+"] not defined correctly.");
            }
        }
    }
    
    public MultiLayerNetworkModelInstance load(InputStream stream) throws Exception
    {
        return new MultiLayerNetworkModelInstance(ModelSerializer.restoreMultiLayerNetworkAndNormalizer(stream, true).getFirst());
    }
}