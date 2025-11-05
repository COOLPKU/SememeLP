from typing import List
import warnings
class Link_List():

    def __init__(self, item:int):
        self.item:int = item
        self.previous:Link_List = None
        self.next:Link_List = None

class Duplicate_Removed_Queue():

    def __init__(self, all_elements_num:int):
        '''
        一个整数的队列(整数元素从0到all_elements_num-1)。
        入队时判断，是否已在队内，若在队内则该元素出队，再从后面入队。
        出队正常从前面出队。
        '''
        self.all_elements_num = all_elements_num
        self.all_elements_node:List[Link_List] = [None]*all_elements_num
        self.head_node = Link_List(0)
        self.tail_node = self.head_node

    def push(self, item:int):
        node = self.all_elements_node[item]
        if node is None:
            node = Link_List(item)
            self.all_elements_node[item] = node
            self.tail_node.next = node
            node.previous = self.tail_node
            self.tail_node = node
            self.head_node.item += 1
        else:
            if self.tail_node is not node:
                node.previous.next = node.next
                node.next.previous = node.previous
                node.next = None
                self.tail_node.next = node
                node.previous = self.tail_node
                self.tail_node = node

    def pop(self):
        if self.head_node is self.tail_node:
            raise IndexError()
        node = self.head_node.next
        if node is self.tail_node:
            self.tail_node = self.head_node
        else:
            node.next.previous = self.head_node
        self.head_node.next = node.next
        self.head_node.item -= 1
        item = node.item
        del node
        self.all_elements_node[item] = None
        return item        
    
    def push_list(self, arr:list):
        for i in arr:
            self.push(i)
    
    def pop_list(self, n:int)->list:
        arr = []
        for i in range(n):
            if len(self)==0:
                warnings.warn('pop_list n>len(queue)')
                break
            arr.append(self.pop())
        return arr
    
    def working(self, max_queue_length:int, arr:list):
        self.push_list(arr)
        if len(self) - max_queue_length>0:
            return self.pop_list(len(self) - max_queue_length)
        else:
            return []

    def get_queue_list(self)->list:
        items = []
        node = self.head_node.next
        while node is not None:
            items.append(node.item)
            node = node.next
        return items
    
    def __len__(self):
        return self.head_node.item

    def __str__(self):
        return f'{self.get_queue_list()}'
