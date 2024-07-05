package au.gov.nla.dnn.training;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import org.json.JSONArray;
import org.json.JSONObject;
import au.gov.nla.dnn.model.ModelInstance;
import au.gov.nla.dnn.record.RawDataRecordProvider;
import au.gov.nla.dnn.sequence.InputSequence;
import au.gov.nla.dnn.sequence.InputSequenceInstance;
import au.gov.nla.dnn.sequence.SequenceDataRecord;
import au.gov.nla.dnn.sequence.SequenceDataRecordProvider;

public class TrainingExecution
{
    public void execute(JSONObject config, String tempDirectory, String saveFile) throws Exception
    {
        Consumer<Exception> errorHandler = new Consumer<Exception>(){
            public void accept(Exception e)
            {
                e.printStackTrace();
            }
        };
        
        JSONObject hyperParamConfig = config.getJSONObject("hyper-parameters");
        long randomSeed;
        
        if(hyperParamConfig.has("random-seed"))
        {
            randomSeed = hyperParamConfig.getLong("random-seed");
        }
        else
        {
            randomSeed = new Random().nextLong();
        }
        
        int evaluationMetric;
        
        switch(hyperParamConfig.getString("evaluation-metric"))
        {
            case "f1":
            {
                evaluationMetric = TrainingHyperParameters.EVALUATION_METRIC_F1;
                break;
            }
            case "precision":
            {
                evaluationMetric = TrainingHyperParameters.EVALUATION_METRIC_PRECISION;
                break;
            }
            case "recall":
            {
                evaluationMetric = TrainingHyperParameters.EVALUATION_METRIC_RECALL;
                break;
            }
            default:
            {
                evaluationMetric = TrainingHyperParameters.EVALUATION_METRIC_F1;
            }
        }
        
        TrainingHyperParameters hyperParameters = new TrainingHyperParameters(
                hyperParamConfig.getInt("max-epochs"), 
                hyperParamConfig.getInt("batch-size"), 
                randomSeed, 
                evaluationMetric);
        
        ModelInstance.Builder<?> modelBuilder = (ModelInstance.Builder<?>)Class.forName(config.getString("model-instance-builder")
                ).getConstructor().newInstance();
        ModelInstance modelInstance = modelBuilder.create(config.getJSONObject("model-instance-builder-config"), randomSeed);
        
        File modelTempDirectoryF = new File(tempDirectory+"/model/");
        File dataTempDirectoryF = new File(tempDirectory+"/data/");
        File dataTempDirectoryFT = new File(tempDirectory+"/data/t/");
        File dataTempDirectoryFE = new File(tempDirectory+"/data/e/");
        
        if(!modelTempDirectoryF.exists())
        {
            modelTempDirectoryF.mkdirs();
        }
        
        deleteDirectory(dataTempDirectoryF);
        
        if(!dataTempDirectoryFT.exists())
        {
            dataTempDirectoryFT.mkdirs();
        }
        if(!dataTempDirectoryFE.exists())
        {
            dataTempDirectoryFE.mkdirs();
        }
        
        List<String> labels = new ArrayList<String>();
        JSONArray labelArray = config.getJSONArray("labels");
        
        for(int i=0; i<labelArray.length(); i++)
        {
            labels.add(labelArray.getString(i));
        }
        
        RawDataRecordProvider trainingRecordProvider = (RawDataRecordProvider)Class.forName(config.getString("training-record-provider")).getConstructor().newInstance();
        trainingRecordProvider.initialise(config.getJSONObject("training-record-provider-config"), errorHandler);
        
        RawDataRecordProvider evaluationRecordProvider = (RawDataRecordProvider)Class.forName(config.getString("evaluation-record-provider")).getConstructor().newInstance();
        evaluationRecordProvider.initialise(config.getJSONObject("evaluation-record-provider-config"), errorHandler);
        
        InputSequence sequence = (InputSequence)Class.forName(config.getString("input-sequence")).getConstructor().newInstance();
        InputSequenceInstance sequenceInstance = sequence.generateInstance(labels, config.getJSONObject("input-sequence-config"));
        
        sequenceInstance.preProcess(randomSeed, trainingRecordProvider);
        trainingRecordProvider.reset();
        
        generateSequenceData(dataTempDirectoryFT, sequenceInstance, trainingRecordProvider);
        generateSequenceData(dataTempDirectoryFE, sequenceInstance, evaluationRecordProvider);
        
        TrainingListener trainingListener = new TrainingListener(){
            public void onEpochComplete(int epoch, int bestEpoch, double bestEpochScore)
            {
                System.out.println("CURRENT EPOCH ["+epoch+"]   BEST EPOCH ["+bestEpoch+"]   BEST SCORE ["+bestEpochScore+"]");
            }
        };
        
        modelInstance.train(hyperParameters, sequenceInstance.getFeatureCount(), labels, modelTempDirectoryF.getAbsolutePath(), 
                trainingListener, 
                errorHandler, 
                createSequenceDataRecordProvider(dataTempDirectoryFT, randomSeed), 
                createSequenceDataRecordProvider(dataTempDirectoryFE, randomSeed));
    }
    
    private SequenceDataRecordProvider createSequenceDataRecordProvider(File directory, long randomSeed) throws Exception
    {
        Random random = new Random(randomSeed);
        List<Integer> indexList = new ArrayList<Integer>();
        ReentrantLock indexLock = new ReentrantLock(true);
        int index = 0;
        
        try(DirectoryStream<Path> stream = Files.newDirectoryStream(directory.toPath()))
        {
            for(@SuppressWarnings("unused") Path entry: stream)
            {
                indexList.add(index);
                index++;
            }
        }
        
        final int totalRecords = indexList.size();
        
        return new SequenceDataRecordProvider()
        {
            private int processedCount;
            
            public void reset() throws Exception
            {
                indexLock.lock();
                
                try
                {
                    processedCount = 0;
                    Collections.shuffle(indexList, random);
                }
                finally
                {
                    indexLock.unlock();
                }
            }
            
            public boolean hasMoreRecords()
            {
                indexLock.lock();
                
                try
                {
                    return processedCount<totalRecords;
                }
                finally
                {
                    indexLock.unlock();
                }
            }
            
            public List<SequenceDataRecord> getMoreRecords(int count) throws Exception
            {
                List<SequenceDataRecord> records = new ArrayList<SequenceDataRecord>();
                
                int batchCount;
                int batchIndex;
                indexLock.lock();
                
                try
                {
                    batchIndex = processedCount;
                    batchCount = Math.min(count, totalRecords-processedCount);
                    processedCount = processedCount+batchCount;
                }
                finally
                {
                    indexLock.unlock();
                }
                
                for(int i=batchIndex; i<batchIndex+batchCount; i++)
                {
                    records.add(readRecordFromDisk(indexList.get(i)));
                }
                
                return records;
            }
            
            public int getTotalRecords()
            {
                return totalRecords;
            }
            
            public void dispose()
            {
                // Nothing to do
            }
            
            private SequenceDataRecord readRecordFromDisk(int index) throws Exception
            {
                try(ObjectInputStream in = new ObjectInputStream(new FileInputStream(new File(directory.getAbsolutePath()+"/"+index))))
                {
                    return (SequenceDataRecord)in.readObject();
                }
            }
        };
    }
    
    private void generateSequenceData(File directory, InputSequenceInstance sequenceInstance, RawDataRecordProvider recordProvider) throws Exception
    {
        int index = 0;
        
        while(recordProvider.hasMoreRecords())
        {
            SequenceDataRecord record = sequenceInstance.process(recordProvider.getNextRecord());
            
            try(ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(new File(directory.getAbsolutePath()+"/"+index))))
            {
                out.writeObject(record);
                out.flush();
            }
        }
    }
    
    private void deleteDirectory(File directory) throws Exception
    {
        try(DirectoryStream<Path> stream = Files.newDirectoryStream(directory.toPath()))
        {
            for(Path entry: stream)
            {
                File file = entry.toFile();
                
                if(file.isDirectory())
                {
                    deleteDirectory(file);
                }
                else
                {
                    entry.toFile().delete();
                }
            }
        }
        
        directory.delete();
    }
}