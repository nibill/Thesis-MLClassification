package Helper;

import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class is for creating a training- and test set based on all your events.
 * It splits all data in 80% training and 20% test
 *
 * @author Thomas Buchegger
 */
public class CreateTrainTest
{
    private final static String WHOLE_TEXTS_PATH = "Data" + File.separator + "Input" + File.separator + "WholeText" + File.separator;
    private final static String WHOLE_TEXTS_PATHMS = "Data" + File.separator + "Input" + File.separator + "WholeTextMS" + File.separator;

    private final static String TRAIN_TEXTS_PATH = "Data" + File.separator + "Input" + File.separator + "TrainTest" + File.separator;
    private final static String TRAIN_TEXTS_PATHMS = "Data" + File.separator + "Input" + File.separator + "TrainTestMS" + File.separator;

    public static void main(String[] args)
    {
        ArrayList<String> shuffledListCat;
        ArrayList<String> shuffledListMS;

        shuffledListCat = FileHandler.GetEventsCat1(WHOLE_TEXTS_PATH + "1");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 1);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 1);

        shuffledListCat = FileHandler.GetEventsCat2(WHOLE_TEXTS_PATH + "2");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 2);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 2);

        shuffledListCat = FileHandler.GetEventsCat3(WHOLE_TEXTS_PATH + "3");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 3);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 3);

        shuffledListCat = FileHandler.GetEventsCat4(WHOLE_TEXTS_PATH + "4");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 4);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 4);

        shuffledListCat = FileHandler.GetEventsCat5(WHOLE_TEXTS_PATH + "5");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 5);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 5);

        shuffledListCat = FileHandler.GetEventsCat6(WHOLE_TEXTS_PATH + "6");
        Collections.shuffle(shuffledListCat);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, true, 6);
        WriteFiles(shuffledListCat, TRAIN_TEXTS_PATH, false, 6);

        shuffledListMS = FileHandler.GetEventsCat5(WHOLE_TEXTS_PATHMS + "1");
        Collections.shuffle(shuffledListMS);
        WriteFiles(shuffledListMS, TRAIN_TEXTS_PATHMS, true, 1);
        WriteFiles(shuffledListMS, TRAIN_TEXTS_PATHMS, false, 1);

        shuffledListMS = FileHandler.GetEventsCat6(WHOLE_TEXTS_PATHMS + "2");
        Collections.shuffle(shuffledListMS);
        WriteFiles(shuffledListMS, TRAIN_TEXTS_PATHMS, true, 2);
        WriteFiles(shuffledListMS, TRAIN_TEXTS_PATHMS, false, 2);

        System.out.println("Create test/train set finished");
    }

    /**
     * Used to save the events back to a file
     *
     * @param eventsToWrite ArrayList with all events to save
     * @param outputBasePath Base path for the train/test folders
     * @param train Do the events belong to train or test
     * @param cat Which category do the events belong to
     */
    private static void WriteFiles(ArrayList<String> eventsToWrite, String outputBasePath, boolean train, int cat)
    {
        Collections.shuffle(eventsToWrite);

        int trainLength = (int) Math.round(eventsToWrite.size() * 0.8);

        List<String> eventsToFile;

        // fill the ArrayList with training or test events
        if (train)
        {
            eventsToFile = eventsToWrite.subList(0, trainLength);
        } else
        {
            eventsToFile = eventsToWrite.subList(trainLength, eventsToWrite.size());
        }

        int i = 1;
        String path = FilenameUtils.concat(outputBasePath, (train ? "train" : "test"));

        for (String event : eventsToFile)
        {
            String catBaseDir = path + File.separator + cat + File.separator + "ct_" + i + ".txt";
            try
            {
                // clear directories
                FileHandler.DeleteFiles(catBaseDir);

                File file = new File(catBaseDir);
                FileWriter fileWriter = new FileWriter(file);
                fileWriter.write(event);
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e)
            {
                e.printStackTrace();
            }
            i++;
        }
    }
}
