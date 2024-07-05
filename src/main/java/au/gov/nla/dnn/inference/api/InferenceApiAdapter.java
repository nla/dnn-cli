package au.gov.nla.dnn.inference.api;

import java.util.Properties;
import java.util.function.Consumer;
import au.gov.nla.dnn.inference.InferenceService;

public interface InferenceApiAdapter
{
    public void initialise(Properties properties, InferenceService service, Consumer<Exception> errorHandler) throws Exception;
    public void dispose() throws Exception;
}
