package au.gov.nla.dnn.record.file;

import java.io.File;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;
import org.json.JSONObject;
import au.gov.nla.dnn.record.RawDataRecord;
import au.gov.nla.dnn.record.RawDataRecordProvider;

public class FileDataRecordProvider implements RawDataRecordProvider
{
    private Consumer<Exception> errorHandler;
    private String directory;
    private DirectoryStream<Path> directoryStream;
    private Iterator<Path> directoryIterator;
    private List<String> labels;
    private int currentLabelIndex;
    
    public void initialise(JSONObject config, Consumer<Exception> errorHandler) throws Exception
    {
        this.errorHandler = errorHandler;
        this.directory = config.getString("directory");
        this.labels = Files.readAllLines(new File(config.getString("labels-file")).toPath());
        
        createStream();
    }
    
    private void createStream() throws Exception
    {
        directoryStream = Files.newDirectoryStream(new File(directory+"/"+labels.get(currentLabelIndex)+"/").toPath());
        directoryIterator = directoryStream.iterator();
    }
    
    public List<String> getLabels()
    {
        return labels;
    }
    
    public void reset()
    {
        currentLabelIndex = 0;
        
        try
        {
            directoryStream.close();
        }
        catch(Exception e)
        {
            errorHandler.accept(e);
        }
        try
        {
            createStream();
        }
        catch(Exception e)
        {
            errorHandler.accept(e);
        }
    }
    
    public boolean hasMoreRecords()
    {
        try
        {
            boolean more = directoryIterator.hasNext();
            
            if(more)
            {
                return true;
            }
            else
            {
                currentLabelIndex++;
                
                if(currentLabelIndex==labels.size())
                {
                    return false;
                }
                else
                {
                    directoryStream.close();
                    createStream();
                    return hasMoreRecords();
                }
            }
        }
        catch(Exception e)
        {
            errorHandler.accept(e);
            return false;
        }
    }
    
    public RawDataRecord getNextRecord() throws Exception
    {
        double[] labelSet = new double[labels.size()];
        labelSet[currentLabelIndex] = 1d;
        
        return new RawDataRecord(Files.readAllBytes(directoryIterator.next()), labelSet);
    }
}
