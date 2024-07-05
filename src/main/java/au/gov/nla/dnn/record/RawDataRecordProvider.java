package au.gov.nla.dnn.record;

import java.util.List;
import java.util.function.Consumer;
import org.json.JSONObject;

public interface RawDataRecordProvider
{
    public void initialise(JSONObject config, Consumer<Exception> errorHandler) throws Exception;
    public List<String> getLabels();
    
    public void reset();
    public boolean hasMoreRecords();
    public RawDataRecord getNextRecord() throws Exception;
}
