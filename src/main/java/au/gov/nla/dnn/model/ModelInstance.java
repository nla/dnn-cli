package au.gov.nla.dnn.model;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.function.Consumer;

import org.json.JSONObject;

import au.gov.nla.dnn.evaluation.ModelEvaluationResult;
import au.gov.nla.dnn.sequence.SequenceDataRecordProvider;
import au.gov.nla.dnn.training.TrainingHyperParameters;
import au.gov.nla.dnn.training.TrainingListener;

public interface ModelInstance
{
    public void train(TrainingHyperParameters hyperParameters, int featureCount, List<String> labels, String tempDirectory, 
            TrainingListener listener, 
            Consumer<Exception> errorHandler,
            SequenceDataRecordProvider trainingRecordProvider, 
            SequenceDataRecordProvider evaluationRecordProvider) throws Exception;
    public ModelEvaluationResult evaluate(SequenceDataRecordProvider evaluationRecordProvider, int batchSize, int featureCount, 
            List<String> labels, Consumer<Exception> errorHandler) throws Exception;
    public double[] infer(double[] features) throws Exception;
    public void save(OutputStream stream) throws Exception;
    
    public interface Builder<T extends ModelInstance>
    {
        public T create(JSONObject config, long randomSeed) throws Exception;
        public T load(InputStream stream) throws Exception;
    }
}