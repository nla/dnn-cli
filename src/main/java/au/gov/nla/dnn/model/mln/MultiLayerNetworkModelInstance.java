package au.gov.nla.dnn.model.mln;

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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
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
                int epoch = epochCounter.incrementAndGet();
                
                try
                {
                    ModelEvaluationResult eval = evaluate(evaluationRecordProvider, hyperParameters.getBatchSize(), featureCount, labels, errorHandler);
                    
                    if(eval.getAccuracy()>bestEpochScore.get())
                    {
                        bestEpochScore.set(eval.getAccuracy());
                        bestEpoch.set(epoch);
                    }
                }
                catch(Exception e)
                {
                    errorHandler.accept(e);
                }
                
                listener.onEpochComplete(epoch, bestEpoch.intValue(), bestEpochScore.doubleValue());
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
        evaluationIterator.reset();
        
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
}
