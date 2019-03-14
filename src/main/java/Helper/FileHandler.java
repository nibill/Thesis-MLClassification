package Helper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;

/**
 * Loads all events from a given directory
 *@author Thomas Buchegger
 */
public class FileHandler
{
    /**
     * Load all evens from a given path to an ArrayList
     * @param path Directory containing the wanted files
     * @return ArrayList with all events
     */
    private static ArrayList<String> LoadEvents(String path)
    {
        ArrayList<String> events = new ArrayList<>();

        File directory = new File(path);
        File[] files = directory.listFiles();

        if(files == null)
        {
            throw new RuntimeException("...");
        }

        for(File file : files)
        {
            try
            {
                String event = new String(Files.readAllBytes(file.toPath()), "UTF-8");
                events.add(event.toLowerCase());
            }
            catch (IOException ex)
            {
                ex.printStackTrace();
            }
        }
        return events;
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat1(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat2(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat3(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat4(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat5(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsCat6(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsSchauen(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @return
     */
    public static ArrayList<String> GetEventsMachen(String path)
    {
        return LoadEvents(path);
    }

    /**
     *
     * @param path
     * @param cat
     * @return
     */
    public static ArrayList<String> GetWeightingWordList(String path, int cat) { return LoadEvents(path + File.separator + cat);}

    /**
     * Delets all files in a given directory
     * @param directoryPath Path with the files to delete
     */
    public static void DeleteFiles(String directoryPath)
    {
        File directory = new File(directoryPath);

        // Get all files in directory
        File[] files = directory.listFiles();

        if(files != null)
        {
            for (File file : files)
            {
                // Delete each file
                if (!file.delete())
                {
                    // Failed to delete file
                    System.out.println("Failed to delete "+file);
                }
            }
        }
    }
}
