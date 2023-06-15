import random

DATA_GENERATION_PROBABILITY = 0.05
MIN_DATA_LIFESPAN = 1
MAX_DATA_LIFESPAN = 50
ESTIMATION_DATA_GENERATION_PROBABILITY = 0.02

class RelaxedDistributeMNIST:
    """
  This class distribute each image among different workers
  It returns a dictionary with key as data owner's id and 
  value as a pointer to the list of data batches at owner's 
  location.
  
  example:-  
  >>> from distribute_data import Distribute_MNIST
  >>> obj = Distribute_MNIST(data_owners= (alice, bob, claire), data_loader= torch.utils.data.DataLoader(trainset)) 
  >>> obj.data_pointer[1]['alice'].shape, obj.data_pointer[1]['bob'].shape, obj.data_pointer[1]['claire'].shape
   (torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 10, 28]))
  """

    def __init__(self, data_owners, data_loader):

        """
         Args:
          data_owners: tuple of data owners
          data_loader: torch.utils.data.DataLoader for MNIST 

        """

        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []
        """
        self.data_pointer:  list of dictionaries where 
        (key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
        example:
        self.data_pointer  = [
                                {"alice": pointer_to_alice_batch1, "bob": pointer_to_bob_batch1},
                                {"alice": pointer_to_alice_batch2, "bob": pointer_to_bob_batch2},
                                ...
                             ]
        """

        self.labels = []
        self.distributed_subdata = []
        self.lifetimes = []
        self.left_out = []

        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:

            curr_data_dict = {}

            # calculate width and height according to the no. of workers for UNIFORM distribution
            height = images.shape[-1]//self.no_of_owner

            self.labels.append(labels)

            # iterate over each worker for distribution of current batch of the self.data_loader
            for i, owner in enumerate(self.data_owners[:-1]):

                # split the image and send it to VirtualWorker (which is supposed to be a dataowner or client)
                image_part_ptr = images[:, :, :, height * i : height * (i + 1)].send(
                    owner
                )

                curr_data_dict[owner.id] = image_part_ptr

            # Repeat same for the remaining part of the image
            last_owner = self.data_owners[-1]
            last_part_ptr = images[:, :, :, height * (i + 1) :].send(last_owner)

            curr_data_dict[last_owner.id] = last_part_ptr

            self.data_pointer.append(curr_data_dict)
            
    def __iter__(self):
        id = 0
        
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (id, data_ptr, label)
            id += 1
            
    def __len__(self):
        
        return len(self.data_loader)-1
            
    def generate_subdata(self):
        self.distributed_subdata = []
        for id, data_ptr, target in self:
            if random.random() <= DATA_GENERATION_PROBABILITY:
                self.distributed_subdata.append((id, data_ptr, target))
                self.lifetimes.append((MAX_DATA_LIFESPAN-MIN_DATA_LIFESPAN)*random.random() + MIN_DATA_LIFESPAN)
            else:
                self.left_out.append((id, data_ptr, target))

    def generate_estimate_subdata(self):
        est_subdata = []
        for id, data_ptr, target in self.distributed_subdata:
            if random.random() <= ESTIMATION_DATA_GENERATION_PROBABILITY:
                est_subdata.append((id, data_ptr, target))
        return est_subdata

    def update_subdata(self):
        removed = []
        idx = 0
        
        for id, data_ptr, target in self.distributed_subdata:
            self.lifetimes[idx] -= 1
            if self.lifetimes[idx] <= 0:
                self.lifetimes.pop(idx)
                self.distributed_subdata.pop(idx)
                self.left_out.append((id, data_ptr, target))
                removed.append((id, data_ptr, target))
                idx -= 1
            idx += 1

        DATA_UPDATE_PROBABILITY = len(removed) / (len(self) - len(self.distributed_subdata))
        added = []
        
        res = True
        while( res ):
            for id, data_ptr, target in self.left_out:
                if random.random() <= DATA_UPDATE_PROBABILITY:
                    self.distributed_subdata.append((id, data_ptr, target))
                    self.lifetimes.append((MAX_DATA_LIFESPAN-MIN_DATA_LIFESPAN)*random.random() + MIN_DATA_LIFESPAN)
                    added.append((id, data_ptr, target))
            res = (len(self.distributed_subdata) == 0)
        return (removed, added)
    
    def split_samples_by_class(self, subdata):
        class_data = {}

        for id, data_ptr, target in subdata:
            if not target.item() in class_data:
                class_data[target.item()] = []
            class_data[target.item()].append((data_ptr, id))
        
        return class_data

    def get_data_ptr_index(self, data_ptr):
        idx = 0
        for _ in range(len(self.data_pointer)):
            shared_items = {owner.id: data_ptr[owner.id] for owner in self.data_owners if owner.id in self.data_pointer and self.data_ptr[owner.id] == self.data_pointer[owner.id] }
            if len(shared_items) > 0:
                break
            idx += 1
        return idx