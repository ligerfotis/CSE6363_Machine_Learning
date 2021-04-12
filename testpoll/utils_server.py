# Prepare username and header and send them We need to encode username to bytes, then count number of bytes and
# prepare header of fixed size, that we encode to bytes as well
import queue
import socket


def send_msg(socket, message, header_length):
    username = message.encode('utf-8')
    username_header = f"{len(username):<{header_length}}".encode('utf-8')
    socket.send(username_header + username)


# Handles message receiving
def receive_file(client_socket, header_length):
    try:

        # Receive our "header" containing message length, it's size is defined and constant
        message_header = client_socket.recv(header_length)

        # If we received no data, client gracefully closed a connection, for example using socket.close() or
        # socket.shutdown(socket.SHUT_RDWR)
        if not len(message_header):
            return False

        # Convert header to int value
        message_length = int(message_header.decode('utf-8').strip())

        # Return an object of message header and message data
        return {'header': message_header, 'data': client_socket.recv(message_length)}

    except:

        # If we are here, client closed connection violently, for example by pressing ctrl+c on his script or just
        # lost his connection socket.close() also invokes socket.shutdown(socket.SHUT_RDWR) what sends information
        # about closing the socket (shutdown read/write) and that's also a cause when we receive an empty message
        return False


def receive_msg(client_socket, header_length):
    # Receive our "header" containing username length, it's size is defined and constant
    msg_header = client_socket.recv(header_length)

    # Convert header to int value
    message_length = int(msg_header.decode('utf-8').strip())

    # Receive and decode username
    message = client_socket.recv(message_length).decode('utf-8')
    return message


def check_username(username, clients):
    for socket in clients:
        if clients[socket]["data"].decode() == username:
            return False
    return True


def connect_client(ip, port):
    # Create a socket socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH,
    # AF_UNIX socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams,
    # socket.SOCK_RAW - raw IP packets
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to a given ip and port
    client_socket.connect((ip, port))

    # Set connection to non-blocking state, so .recv() call won;t block, just return some exception we'll handle
    client_socket.setblocking(False)
    return client_socket


def spelling_check(file_to_checked, username, lexicon_file):
    """
    Checks a file for spelling errors against lexicon.txt.
    ASSUMPTION: There is only one period at the end of each line.
    :param username: username string
    :param file_to_checked: file name to be spelling-checked
    :return: a file with [] around the misspelled word
    """
    # open the lexicon file
    with open('server_files/{}'.format(lexicon_file)) as lex_file:
        lex_array = lex_file.readline().split(" ")

    # open the file to be checked
    with open(file_to_checked) as file:
        text = file.readlines()

    array_checked_text = []
    # array of lower cased lexicon words
    lower_lex_array = [word.lower() for word in lex_array]
    # array of upper cased lexicon words
    upper_lex_array = [word.upper() for word in lex_array]
    # array of first letter upper cased and the rest lower cased lexicon words
    cap_lex_array = [word.capitalize() for word in lower_lex_array]
    for line in text:
        # convert string to array of words while removing periods, commas and next line special chars
        line_array = line.strip("\n.,").split(" ")
        # substitute a word in the word array if it is in the lower/upper/capitalized lexicon word array
        corrected_array = ['[' + word_i + ']'
                           if word_i in lower_lex_array or word_i in upper_lex_array or word_i in cap_lex_array
                           else word_i
                           for word_i in line_array]

        array_checked_text.append(corrected_array)

    # convert word arrays into strings and add period at the end
    string_checked_text = []
    for line in array_checked_text:
        string_checked_text.append(" ".join(line) + '.\n')

    # write the annotated text to a file
    checked_file_name = "server_files/checked_text_{}.txt".format(username)
    with open(checked_file_name, 'w') as chked_file:
        chked_file.writelines(string_checked_text)

    return checked_file_name


def q_polling(clients, header_length):
    polls = {}
    for client_socket in clients:
        # poll each client
        send_msg(client_socket, "poll", header_length)
        q = queue.Queue()
        while 1:
            poll_msg = receive_msg(client_socket, header_length)
            if poll_msg == 'poll_end':
                polls[client_socket] = q
                break
            else:
                q.put(poll_msg)
    return polls


def print_dict_queues(q_dict):
    for q in q_dict.values():
        q_size = q.qsize()
        for _ in range(q_size):
            tmp = q.get()
            print(tmp)
            q.put(tmp)
    return q_dict

