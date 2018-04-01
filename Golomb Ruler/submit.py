# do not modify the function names
# You are given L and M as input
# Each of your functions should return the minimum possible L value alongside the marker positions
# Or return -1,[] if no solution exists for the given L

# The complete code is based on the pseudocode in the Russell-Norvig book - Fig. 6.5
# Your backtracking function implementation
import time
import copy

def BT(L, M):
    "*** YOUR CODE HERE ***"
    consistencies = {}
    #global nodes_counter
    #nodes_counter = 0

    # Create a domain which is a 2D array
    domain = [[1 for x in range(L + 1)] for x in range(M)]
    assignment = {}
    for i in range(M):
        assignment[i] = -1

    # Create domain values
    def order_domain_values(var):
        new_domain_values = []
        for i in range(len(domain[var])):
            if domain[var][i] != 0:
                new_domain_values.append(i)
        return new_domain_values

    # Assign values to the domain
    def assign_values(var, val, current_assignment):
        for key, value in current_assignment.items():
            if key == var:
                current_assignment[key] = val

    def consistent(var, value, assignment):
        # Check if the value to be assigned will result in a correct solution or not
        assign_values(var, value, assignment)

        current_assignment_status = tuple(assignment.items())
        if current_assignment_status in consistencies:
            return consistencies[current_assignment_status]

        for assmnt in range(1, len(assignment)):
            if (assignment[assmnt] <= assignment[assmnt - 1]) \
                    and assignment[assmnt] != -1 \
                    and assignment[assmnt - 1] != -1:
                consistencies[current_assignment_status] = False
                return False
        ret = distances(assignment)
        if ret == 1:
            consistencies[current_assignment_status] = True
            return 1

    def distances(assignment):
        valList = assignment.values()
        disList = []
        for i in range(len(valList)):
            if valList[i] == -1:
                continue
            for j in range(i + 1, len(valList)):
                if valList[j] == -1: continue
                if abs(valList[j] - valList[i]) not in disList:
                    disList.append(abs(valList[j] - valList[i]))
                else:
                    return 0
        return 1
    def backtrack(assignment):
        assignment_complete = False
        for i in range(M):
            if assignment[i] == -1:
                assignment_complete = False
                break
            else:
                assignment_complete = True

        if assignment_complete:
            if consistent(-1, -1, assignment):
                # Return a solution or failure
                return assignment

        var = 0
        for i in range(M):
            if assignment[i] == -1:
                var = i
                break

        for value in order_domain_values(var):
            #global nodes_counter
            #nodes_counter = nodes_counter + 1
            if consistent(var, value, assignment):
                assignment[var] = value
                result = backtrack(assignment)
                if result:
                    return result
                assignment[var] = -1

        return False

    simple_bt_result = backtrack(assignment)
    if simple_bt_result is False:
        return -1, []

    marks = simple_bt_result.values()
    return marks[-1], marks


# Your backtracking+Forward checking function implementation
def FC(L, M):
    "*** YOUR CODE HERE ***"
    consistencies = {}
    #global nodes_counter
    #nodes_counter = 0
    assignment = {}

    domain = [[1 for x in range(L + 1)] for x in range(M)]
    for i in range(M):
        assignment[i] = -1

    def order_domain_values(var):
        possible_values = []
        if var is not None:
            for i in range(len(domain[var])):
                if domain[var][i] != 0:
                    possible_values.append(i)
        return possible_values

    def assign_values(var, value, current_assignment):
        assignment = {}
        for i in range(0, len(current_assignment)):
            if i == var:
                assignment[i] = value
            else:
                assignment[i] = current_assignment[i]
        return assignment

    # Consistency check
    def consistent(var, value, current_assignment):
        # Check if the value to be assigned will result in a correct solution or not
        assignment = assign_values(var, value, current_assignment)

        current_assignment_status = tuple(assignment.items())
        if current_assignment_status in consistencies:
            return consistencies[current_assignment_status]

        for assmnt in range(1, len(assignment)):
            if (assignment[assmnt] <= assignment[assmnt - 1]) \
                    and assignment[assmnt] != -1 \
                    and assignment[assmnt - 1] != -1:
                consistencies[current_assignment_status] = False
                return False

        ret = distances(assignment)
        if ret == 1:
            consistencies[current_assignment_status] = True
            return 1

    # Check the distance consistencies
    def distances(assignment):
        valList = assignment.values()
        disList = []
        for i in range(len(valList)):
            if valList[i] == -1:
                continue
            for j in range(i + 1, len(valList)):
                if valList[j] == -1: continue
                if abs(valList[j] - valList[i]) not in disList:
                    disList.append(abs(valList[j] - valList[i]))
                else:
                    return 0
        return 1

    # Infer if a solution exists or if it's a failure while forward checking
    def inference(var):
        set_infs = []
        for i in range(0,len(domain)):
            # if an assignment exists
            if i == var:
                continue
            if assignment[i] != -1:
                continue
            for j in range(0,len(domain[i])):
                if domain[i][j] == 0:
                    continue
                if not (consistent(i, j, assignment)):
                    domain[i][j] = 0
            if domain[i].count(1) == 0:
                # No more values to give
                return -1
            if domain[i].count(1) == 1:
                # One value left, just get that one
                set_infs.append((i, domain[i][1]))
        return set_infs

    def backup_domains(src, dest):
        for i in range(len(dest)):
            for j in range(len(dest[i])):
                src[i][j] = dest[i][j]

    # Driver function for backtracking
    def backtrack(assignment):
        for i in range(M):
            if assignment[i] == -1:
                assignment_complete = False
                break
            else:
                assignment_complete = True

        if assignment_complete:
            if consistent(-1, -1, assignment):
                # Return a solution or failure
                return assignment

        # Select unassigned variable from the domain
        var = 0
        for i in range(M):
            if assignment[i] == -1:
                var = i
                break

        for value in order_domain_values(var):
            #global nodes_counter
            #nodes_counter = nodes_counter + 1
            if consistent(var, value, assignment):
                assignment[var] = value
                domain1 = copy.deepcopy(domain)

                #Logic to check if value assignments made by inference lead to a failure
                #If they do, they're made unaccessible
                if inference(var) != -1:
                    for i in inference(var):
                        assignment[i[0]] = -1
                    if backtrack(assignment):
                        return backtrack(assignment)
                assignment[var] = -1
                # Copy back the domain from the backup
                backup_domains(domain, domain1)

        return False

    fc_result = backtrack(assignment)
    if fc_result is False:
        return -1, []

    marks = fc_result.values()
    return L, marks


# Bonus: backtracking + constraint propagation
def CP(L, M):
    "*** YOUR CODE HERE ***"
    return -1, []


time1 = time.time()
answer = FC(6, 4)
time2 = time.time()
print 'FC function took %0.3f ms' % ((time2 - time1) * 1000.0)
print "final answer= ", answer

time1 = time.time()
answer = BT(6, 4)
time2 = time.time()
print 'BT function took %0.3f ms' % ((time2 - time1) * 1000.0)
print "final answer= ", answer