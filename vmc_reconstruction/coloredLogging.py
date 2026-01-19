import logging
# from colorama import Fore, Back, Style

TODO = 60

# colorMap = {
#     logging.DEBUG: lambda s: Fore.BLUE + s + Fore.RESET,
#     logging.INFO: lambda s: Fore.WHITE + s + Fore.RESET,
#     logging.WARNING: lambda s: Fore.YELLOW + s + Fore.RESET,
#     logging.ERROR: lambda s: Fore.RED + s + Fore.RESET,
#     logging.CRITICAL: lambda s: Back.RED + Fore.WHITE + Style.BRIGHT + s + Style.RESET_ALL,
#     TODO: lambda s: Back.YELLOW + Fore.BLACK + Style.BRIGHT + s + Style.RESET_ALL,
# }


class ColoredStreamHandler(logging.StreamHandler):
    # def __init__(self, *args, **kwargs):
    #     self.colorMap = kwargs.pop('colorMap', colorMap)
    #     super().__init__(*args, **kwargs)

    # def emit(self, record):
    #     try:
    #         message = self.format(record)
    #         stream = self.stream
    #         stream.write(message)
    #         stream.write(getattr(self, 'terminator', '\n'))
    #         self.flush()
    #     except (KeyboardInterrupt, SystemExit):
    #         raise
    #     except:
    #         self.handleError(record)

    # def colorize(self, message, record):
    #     if record.levelno in self.colorMap:
    #         message = self.colorMap[record.levelno](message)
    #     return message

    # def format(self, record):
    #     message = logging.StreamHandler.format(self, record)
    #     if hasattr(self.stream, 'isatty') and self.stream.isatty():
    #         # Don't colorize any traceback
    #         parts = message.split('\n', 1)
    #         parts[0] = self.colorize(parts[0], record)
    #         message = '\n'.join(parts)
    #     return message
    pass


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ColoredStreamHandler())
    logging.debug('DEBUG')
    logging.info('INFO')
    logging.warning('WARNING')
    logging.error('ERROR')
    logging.critical('CRITICAL')
