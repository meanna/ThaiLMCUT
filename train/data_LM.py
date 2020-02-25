import torch

# define vocabulary
thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ'
thai_chars = thai_chars + "."
itos = ['<START>'] + ['<END>'] + ['<UNK>'] + list(thai_chars)
stoi = {k:v for k,v in enumerate(itos)}

cuda = torch.cuda.is_available()

# create batches
def prepareDatasetChunks(args, data):
    count = 0
    print("creating batches...")
    numerified = []
    for chunk in data:
        for char in chunk:
            if char == " " or char == "":
                continue
            count += 1
            numerified.append(stoi[char] if char in stoi else 2)

        cutoff = int(len(numerified) / (args.batchSize * args.sequence_length)) * (
                args.batchSize * args.sequence_length)

        numerifiedCurrent = numerified[:cutoff]
        numerified = numerified[cutoff:]
        if cuda:

            numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1,
                                                                         args.sequence_length).transpose(0,
                                                                                                         1).transpose(1,
                                                                                                                      2).cuda()
        else:
            numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1,
                                                                         args.sequence_length).transpose(0,
                                                                                                         1).transpose(
                1, 2)

        numberOfSequences = numerifiedCurrent.size()[0]
        for i in range(numberOfSequences):
            yield numerifiedCurrent[i]
