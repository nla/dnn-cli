package au.gov.nla.dnn.sequence.text.tfidf;

import java.util.ArrayList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;
import au.gov.nla.dnn.sequence.InputSequence;
import au.gov.nla.dnn.sequence.InputSequenceInstance;

public class TFIDFVectorSpaceInputSequence implements InputSequence
{
    public InputSequenceInstance generateInstance(List<String> labels, JSONObject config) throws Exception
    {
        List<String> exclusionPatterns = new ArrayList<String>();
        JSONArray array = config.getJSONArray("word-exclusion-patterns");
        
        for(int i=0; i<array.length(); i++)
        {
            exclusionPatterns.add(array.getString(i));
        }
        
        return new TFIDFVectorSpaceInputSequenceInstance(
                config.getString("word-split-pattern"), 
                exclusionPatterns,
                config.getInt("word-min-characters"), 
                config.getInt("word-max-characters"), 
                config.getInt("word-min-occurrances"), 
                labels);
    }
}
