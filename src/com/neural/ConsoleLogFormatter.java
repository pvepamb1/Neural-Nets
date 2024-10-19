package com.neural;

import java.util.logging.Formatter;
import java.util.logging.LogRecord;

public class ConsoleLogFormatter extends Formatter
{
    @Override
    public String format(LogRecord record)
    {
        // Custom format: [LEVEL] Timestamp - LoggerName: message
        return String.format("[%s] %tF %<tT - %s: %s%n",
                record.getLevel(),
                new java.util.Date(record.getMillis()),
                record.getLoggerName(),
                record.getMessage());
    }
}
