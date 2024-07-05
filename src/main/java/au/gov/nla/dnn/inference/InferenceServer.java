package au.gov.nla.dnn.inference;

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import org.json.JSONArray;
import org.json.JSONObject;
import au.gov.nla.dnn.inference.api.InferenceApiAdapter;
import au.gov.nla.dnn.model.ModelInstance;
import au.gov.nla.dnn.model.SerializedModelState;
import au.gov.nla.dnn.record.RawDataRecord;
import au.gov.nla.dnn.sequence.InputSequenceInstance;

public class InferenceServer
{
    private HashMap<String, InferenceModel> models;
    private List<InferenceApiAdapter> apiAdapters;
    
    public void start(JSONObject serverConfig) throws Exception
    {
        // Set up shutdown hook
        
        Consumer<Exception> errorHandler = new Consumer<Exception>(){
            public void accept(Exception e)
            {
                e.printStackTrace();
            }
        };
        
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable(){
            public void run()
            {
                for(InferenceApiAdapter adapter: apiAdapters)
                {
                    try
                    {
                        adapter.dispose();
                    }
                    catch(Exception e)
                    {
                        errorHandler.accept(e);
                    }
                }
            }
        }));
        
        // Load configuration
        
        JSONArray adapterListConfig = serverConfig.getJSONArray("api-adapters");
        JSONArray modelListConfig = serverConfig.getJSONArray("models");
        
        // Create Inference Service
        
        InferenceService inferenceService = new InferenceService(){
            public InferenceResult infer(String modelId, byte[] record) throws Exception
            {
                InferenceModel model = models.get(modelId);
                
                if(model==null)
                {
                    throw new Exception("Model ["+modelId+"] is not loaded.");
                }
                
                model.getLock().lock();
                double[] result;
                
                try
                {
                    result = model.getModel().infer(model.getSequence().process(
                            new RawDataRecord(record, new double[model.getLabels().length])).getFeatures());
                }
                finally
                {
                    model.getLock().unlock();
                }
                
                int highest = 0;
                double highestValue = 0d;
                HashMap<String, Double> scores = new HashMap<String, Double>();
                
                for(int i=0; i<result.length; i++)
                {
                    double value = result[i];
                    scores.put(model.getLabels()[i], value);
                    
                    if(value>highestValue)
                    {
                        highestValue = value;
                        highest = i;
                    }
                }
                
                return new InferenceResult(model.getLabels()[highest], highestValue, scores);
            }
        };
        
        // Load models
        
        for(int i=0; i<modelListConfig.length(); i++)
        {
            JSONObject modelEntry = modelListConfig.getJSONObject(i);
            models.put(modelEntry.getString("id"), loadModel(modelEntry.getString("file")));
        }
        
        // Initialise API adapters
        
        for(int i=0; i<adapterListConfig.length(); i++)
        {
            JSONObject adapterEntry = adapterListConfig.getJSONObject(i);
            InferenceApiAdapter adapter = (InferenceApiAdapter)Class.forName(adapterEntry.getString("class")).getConstructor().newInstance();
            Properties properties = new Properties();
            
            try(FileReader reader = new FileReader(adapterEntry.getString("properties-file")))
            {
                properties.load(reader);
            }
            
            adapter.initialise(properties, inferenceService, errorHandler);
        }
    }
    
    private InferenceModel loadModel(String path) throws Exception
    {
        try(ObjectInputStream in = new ObjectInputStream(new FileInputStream(path)))
        {
            SerializedModelState state = (SerializedModelState)in.readObject();
            
            try(ByteArrayInputStream bytes = new ByteArrayInputStream(state.getModel()))
            {
                ModelInstance instance = ((ModelInstance.Builder<?>)Class.forName(state.getBuilderClass()).getConstructor().newInstance()).load(bytes);
                return new InferenceModel(instance, state.getSequenceInstance(), state.getLabels());
            }
        }
    }
    
    public class InferenceModel
    {
        private ModelInstance model;
        private InputSequenceInstance sequence;
        private String[] labels;
        private ReentrantLock lock;
        
        public InferenceModel(ModelInstance model, InputSequenceInstance sequence, String[] labels)
        {
            this.model = model;
            this.sequence = sequence;
            this.labels = labels;
            this.lock = new ReentrantLock(true);
        }

        public ModelInstance getModel()
        {
            return model;
        }

        public InputSequenceInstance getSequence()
        {
            return sequence;
        }
        
        public String[] getLabels()
        {
            return labels;
        }

        public ReentrantLock getLock()
        {
            return lock;
        }
    }
}