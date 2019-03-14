package NNApproach;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.*;

/**
 * Create datasets
 * @author Alex Black
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/sentenceclassification/CnnSentenceClassificationExample.java
 * @author Thomas Buchegger
 */
public class DataSetStuff
{
    /**
     * Create datasets
     *
     * @param baseTrainPath     Path to train/test events
     * @param isTraining        Should a train or test dataset be created
     * @param wordVectors       Path to the pre-trained wordvectors
     * @param minibatchSize     Batchsize
     * @param maxSentenceLength Max. length of sentence
     * @param rng               Range of rng
     * @param cat               Do you want to create datasets for categories or machen/schauen
     * @return Train or test dataset
     */
    public static DataSetIterator IterateDataSet(String baseTrainPath, boolean isTraining, WordVectors wordVectors, int minibatchSize, int maxSentenceLength, Random rng, boolean cat)
    {
        String path = FilenameUtils.concat(baseTrainPath, (isTraining ? "train/" : "test/"));
        Map<String, List<File>> categoriesFileMap = new HashMap<>();

        if (cat)
        {
            String cat1BaseDir = FilenameUtils.concat(path, "1");
            String cat2BaseDir = FilenameUtils.concat(path, "2");
            String cat3BaseDir = FilenameUtils.concat(path, "3");
            String cat4BaseDir = FilenameUtils.concat(path, "4");
            String cat5BaseDir = FilenameUtils.concat(path, "5");
            String cat6BaseDir = FilenameUtils.concat(path, "6");

            File cat1 = new File(cat1BaseDir);
            File cat2 = new File(cat2BaseDir);
            File cat3 = new File(cat3BaseDir);
            File cat4 = new File(cat4BaseDir);
            File cat5 = new File(cat5BaseDir);
            File cat6 = new File(cat6BaseDir);

            categoriesFileMap.put("sport", Arrays.asList(cat1.listFiles()));
            categoriesFileMap.put("politik", Arrays.asList(cat2.listFiles()));
            categoriesFileMap.put("kultur", Arrays.asList(cat3.listFiles()));
            categoriesFileMap.put("musik", Arrays.asList(cat4.listFiles()));
            categoriesFileMap.put("essen", Arrays.asList(cat5.listFiles()));
            categoriesFileMap.put("freizeit", Arrays.asList(cat6.listFiles()));
        }
        else
        {
            String machenBaseDir = FilenameUtils.concat(path, "1");
            String schauenBaseDir = FilenameUtils.concat(path, "2");

            File machen = new File(machenBaseDir);
            File schauen = new File(schauenBaseDir);

            categoriesFileMap.put("machen", Arrays.asList(machen.listFiles()));
            categoriesFileMap.put("schauen", Arrays.asList(schauen.listFiles()));
        }

        LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(categoriesFileMap, rng);

        return new CnnSentenceDataSetIterator.Builder()
            .sentenceProvider(sentenceProvider)
            .wordVectors(wordVectors)
            .minibatchSize(minibatchSize)
            .maxSentenceLength(maxSentenceLength)
            .useNormalizedWordVectors(false)
            .build();
    }
}
