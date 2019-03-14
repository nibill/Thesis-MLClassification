package Helper;

import java.io.*;
import java.util.ArrayList;

/**
 * Merge events to one file
 *
 * @author Thomas Buchegger
 */
public class Merger
{
    /**
     * Merge all events from a given ArrayList to a single file
     *
     * @param eventList  List containing all events
     * @param outputPath Path where the file should be saved
     */
    public static void MergeTexts(ArrayList<String> eventList, String outputPath, int cat) throws IOException
    {
        File file = new File(outputPath, "merge" + cat + ".txt");

        // firstly empty already existing files
        EmptyFile(file.getAbsolutePath());

        ArrayList<String> preprocessedEvents = TextPreprocessing.DoPreprocessing(eventList);

        for (String event : preprocessedEvents)
        {
            if (file.exists())
            {
                try (PrintWriter pw = new PrintWriter(new FileOutputStream(file.getAbsolutePath(), true)))
                {
                    pw.println(event);
                } catch (FileNotFoundException e)
                {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Empties the file
     *
     * @param filePath Path to file which should be emptied
     */
    private static void EmptyFile(String filePath)
    {
        try
        {
            PrintWriter pw = new PrintWriter(filePath);
            pw.print("");
            pw.close();
        } catch (IOException e)
        {
            System.out.println("Invalid permissions.");
        }
    }
}
