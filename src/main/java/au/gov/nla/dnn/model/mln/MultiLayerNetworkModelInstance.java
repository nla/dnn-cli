package au.gov.nla.dnn.model.mln;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
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
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
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
import au.gov.nla.dnn.evaluation.MetricEvaluationScoreCalculator;
import au.gov.nla.dnn.evaluation.ModelEvaluationResult;
import au.gov.nla.dnn.evaluation.ModelEvaluationResult.ConfusionStatistics;
import au.gov.nla.dnn.model.ModelInstance;
import au.gov.nla.dnn.sequence.SequenceDataRecord;
import au.gov.nla.dnn.sequence.SequenceDataRecordProvider;
import au.gov.nla.dnn.training.TrainingHyperParameters;
import au.gov.nla.dnn.training.TrainingListener;

public class MultiLayerNetworkModelInstance implements ModelInstance
{
    private MultiLayerNetwork network;
    
    public MultiLayerNetworkModelInstance(MultiLayerNetwork network)
    {
        this.network = network;
    }
    
    public void train(TrainingHyperParameters hyperParameters, int featureCount, List<String> labels, String tempDirectory, 
            TrainingListener listener, 
            Consumer<Exception> errorHandler,
            SequenceDataRecordProvider trainingRecordProvider, 
            SequenceDataRecordProvider evaluationRecordProvider) throws Exception
    {
        AtomicInteger epochCounter = new AtomicInteger();
        AtomicInteger bestEpoch = new AtomicInteger();
        AtomicDouble bestEpochScore = new AtomicDouble();
        
        network.init();
        network.addListeners(new org.deeplearning4j.optimize.api.TrainingListener(){
            public void iterationDone(Model model, int iteration, int epoch){}
            public void onEpochStart(Model model){}
            public void onForwardPass(Model model, List<INDArray> activations){}
            public void onForwardPass(Model model, Map<String, INDArray> activations){}
            public void onGradientCalculation(Model model){}
            public void onBackwardPass(Model model){}
            public void onEpochEnd(Model model)
            {
                listener.onEpochComplete(epochCounter.incrementAndGet(), bestEpoch.intValue(), bestEpochScore.doubleValue());
            }
        });
        
        AsyncDataSetIterator trainingIterator = new AsyncDataSetIterator(new SequenceDataRecordIterator(hyperParameters.getBatchSize(), featureCount, labels, trainingRecordProvider, errorHandler), 3, true);
        AsyncDataSetIterator evaluationIterator = new AsyncDataSetIterator(new SequenceDataRecordIterator(hyperParameters.getBatchSize(), featureCount, labels, evaluationRecordProvider, errorHandler), 3, true);
        
        MetricEvaluationScoreCalculator scoreCalculator = new MetricEvaluationScoreCalculator(hyperParameters, labels.size(), evaluationIterator);
        
        EarlyStoppingConfiguration<MultiLayerNetwork> config = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(hyperParameters.getMaxEpochs()))
                .scoreCalculator(scoreCalculator)
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(tempDirectory))
                .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(config, network, trainingIterator);
        
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        
        if(result.getTerminationReason().equals(EarlyStoppingResult.TerminationReason.Error))
        {
            throw new Exception("An exception occurred while training: "+result.getTerminationDetails());
        }
        
        network = result.getBestModel();
        
        System.gc();
        Nd4j.getMemoryManager().invokeGc();
    }
    
    public ModelEvaluationResult evaluate(SequenceDataRecordProvider evaluationRecordProvider, int batchSize, int featureCount, 
            List<String> labels, Consumer<Exception> errorHandler) throws Exception
    {
        AsyncDataSetIterator evaluationIterator = new AsyncDataSetIterator(new SequenceDataRecordIterator(batchSize, featureCount, labels, evaluationRecordProvider, errorHandler), 3, true);
        
        LinkedHashMap<String, Double> accuracyRatings = new LinkedHashMap<String, Double>();
        LinkedHashMap<String, ConfusionStatistics> confusionMatrix = new LinkedHashMap<String, ConfusionStatistics>();
        
        Evaluation eval = new Evaluation(labels.size());
        
        while(evaluationIterator.hasNext())
        {
            DataSet set = evaluationIterator.next();
            List<String> meta = set.getExampleMetaData(String.class);
            INDArray out = network.output(set.getFeatures(), false);
            eval.eval(set.getLabels(), out, meta);
        }
        
        double totalFp = 0;
        double totalFn = 0;
        double totalTp = 0;
        double totalWeightedMcc = 0d;
        
        for(int i=0; i<labels.size(); i++)
        {
            String label = labels.get(i);
            
            int fp = eval.falsePositives().get(i);
            int fn = eval.falseNegatives().get(i);
            int tp = eval.truePositives().get(i);
            int tn = eval.trueNegatives().get(i);
            
            double weightedMcc = eval.matthewsCorrelation(i)*((double)(tp+fn));
            totalWeightedMcc = totalWeightedMcc+weightedMcc;
            
            totalFp = totalFp+fp;
            totalFn = totalFn+fn;
            totalTp = totalTp+tp;
            double fdr = ((double)fp)/((double)(tp+fp));
            
            accuracyRatings.put(label, eval.f1(i));
            confusionMatrix.put(label, new ModelEvaluationResult.ConfusionStatistics(tp, tn, fp, fn, 
                    eval.precision(i), 
                    eval.recall(i), 
                    fdr, 
                    eval.falsePositiveRate(i),
                    totalWeightedMcc));
        }
        
        totalWeightedMcc = totalWeightedMcc/(double)evaluationRecordProvider.getTotalRecords();
        
        for(String label: labels)
        {
            confusionMatrix.get(label).setMcc(totalWeightedMcc);
        }
        
        double precision = totalTp/(totalTp/totalFp);
        double recall = totalTp/(totalTp/totalFn);
        double f1 = (2d*precision*recall)/(precision+recall);
        
        return new ModelEvaluationResult(f1, accuracyRatings, confusionMatrix);
    }
    
    public double[] infer(double[] features) throws Exception
    {
        INDArray featureArray = Nd4j.create(1, features.length);
        
        for(int i=0; i<features.length; i++)
        {
            featureArray.putScalar(new int[]{0, i}, features[i]);
        }
        
        INDArray prediction = network.output(featureArray, false);
        return new double[(int)prediction.length()];
    }

    public void save(OutputStream stream) throws Exception
    {
        ModelSerializer.writeModel(network, stream, true);
    }
    
    public static class SequenceDataRecordIterator implements DataSetIterator
    {
        private static final long serialVersionUID = 1L;
        
        private int batchSize;
        private int featureLength;
        private List<String> labels;
        private SequenceDataRecordProvider recordProvider;
        private DataSetPreProcessor preProcessor;
        private Consumer<Exception> errorHandler;
        
        public SequenceDataRecordIterator(int batchSize, int featureLength, List<String> labels, SequenceDataRecordProvider recordProvider, Consumer<Exception> errorHandler)
        {
            this.batchSize = batchSize;
            this.featureLength = featureLength;
            this.labels = labels;
            this.recordProvider = recordProvider;
            this.errorHandler = errorHandler;
        }
        
        public boolean hasNext()
        {
            return recordProvider.hasMoreRecords();
        }
        
        public DataSet next()
        {
            return next(batchSize);
        }
        
        public boolean asyncSupported()
        {
            return true;
        }
        
        public int batch()
        {
            return batchSize;
        }
        
        public List<String> getLabels()
        {
            return labels;
        }
        
        public DataSetPreProcessor getPreProcessor()
        {
            return preProcessor;
        }
        
        public int inputColumns()
        {
            return featureLength;
        }
        
        public DataSet next(int batch)
        {
            try
            {
                List<SequenceDataRecord> records = recordProvider.getMoreRecords(batch);
                int exampleCount = Math.max(records.size(), 2); // if 1 example, ND4J assumes the array is a different rank
                
                INDArray featureArray = Nd4j.create(exampleCount, featureLength);
                INDArray labelArray = Nd4j.create(exampleCount, labels.size());
                INDArray featureMaskArray = Nd4j.zeros(exampleCount, 1);
                INDArray labelMaskArray = Nd4j.zeros(exampleCount, 1);
                
                for(int i=0; i<records.size(); i++)
                {
                    SequenceDataRecord record = records.get(i);
                    
                    for(int f=0; f<featureLength; f++)
                    {
                        featureArray.putScalar(new int[]{i, f}, record.getFeatures()[f]);
                    }
                    
                    featureMaskArray.putScalar(new int[]{i}, 1d);
                    boolean anyLabel = false;
                    
                    for(int l=0; l<labels.size(); l++)
                    {
                        double label = record.getLabels()[l];
                        labelArray.putScalar(new int[]{i, l}, record.getLabels()[l]);
                        
                        if(!anyLabel && label>0d)
                        {
                            anyLabel = true;
                        }
                    }
                    if(anyLabel)
                    {
                        labelMaskArray.putScalar(new int[]{i}, 1d);
                    }
                }
                
                return new DataSet(featureArray, labelArray, featureMaskArray, labelMaskArray);
            }
            catch(Exception e)
            {
                errorHandler.accept(e);
                return new DataSet();
            }
        }
        
        public void reset()
        {
            try
            {
                recordProvider.reset();
            }
            catch(Exception e)
            {
                errorHandler.accept(e);
            }
        }
        
        public boolean resetSupported()
        {
            return true;
        }
        
        public void setPreProcessor(DataSetPreProcessor preProcessor)
        {
            this.preProcessor = preProcessor;
        }
        
        public int totalOutcomes()
        {
            return labels.size();
        }
    }
    
    public static class Builder implements ModelInstance.Builder<MultiLayerNetworkModelInstance>
    {
        public MultiLayerNetworkModelInstance create(JSONObject config, long randomSeed) throws Exception
        {
            NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                    .weightInit(getWeightInit(config.getString("weight-init")))
                    .activation(getActivationFunction(config.getString("activation-function")))
                    .optimizationAlgo(getOptimizationAlgorithm(config.getString("optimization-algorithm")))
                    .updater(getUpdater(config.getString("updater"), config.getJSONObject("updater-params")))
                    .seed(randomSeed);
            
            NeuralNetConfiguration.ListBuilder definitionBuilder = b.list();
            int inputNodes = config.getInt("input-node-count");
            int outputNodes = config.getInt("output-node-count");
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
}
