package au.gov.nla.dnn.sequence.text.tfidf;

import java.util.List;
import org.json.JSONObject;
import au.gov.nla.dnn.sequence.InputSequence;
import au.gov.nla.dnn.sequence.InputSequenceInstance;

public class TFIDFVectorSpaceInputSequence implements InputSequence
{
    public InputSequenceInstance generateInstance(List<String> labels, JSONObject config) throws Exception
    {
        return new TFIDFVectorSpaceInputSequenceInstance(
                config.getInt("word-min-characters"), 
                config.getInt("word-min-occurrances"), 
                labels);
    }
}