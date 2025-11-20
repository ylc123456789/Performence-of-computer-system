# Simulation Topology
#              n1                  n5
#               \                  /
#   4000Mb,500ms \   1000Mb,50ms  / 4000Mb,500ms
#              n3 --------------- n4
#   4000Mb,800ms /                \ 4000Mb,800ms
#               /                  \
#             n2                   n6 

set ns [new Simulator]

# Try to get a random seed from the environment variable 'SEED'.
set seed [expr {[info exists ::env(SEED)] ? $::env(SEED) : [clock seconds]}]
puts "Current random seed: $seed"

# Try to get the bottleneck link bandwidth from the environment variable 'BW'.
set bw [expr {[info exists ::env(BW)] ? $::env(BW) : "1000Mb"}]

$ns color 1 Blue
$ns color 2 Red

set namfile [open yeah.nam w]
$ns namtrace-all $namfile

set tracefile1 [open yeahTrace.tr w]
$ns trace-all $tracefile1

proc finish {} {
    global ns namfile tracefile1
    $ns flush-trace
    close $namfile
    close $tracefile1
    exit 0
}

# Create all the nodes required for the topology
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

# Randomize some topology parameters based on the seed to introduce variability
set bottleneck_delay [expr {50 + ($seed % 41) - 20}]
set queue_limit [expr {10 + ($seed % 11) - 5}]

# Create duplex links between nodes with specified bandwidth, delay, and queue type
$ns duplex-link $n1 $n3 4000Mb 500ms RED
$ns duplex-link $n2 $n3 4000Mb 800ms RED 
$ns duplex-link $n3 $n4 $bw ${bottleneck_delay}ms RED
$ns duplex-link $n4 $n5 4000Mb 500ms RED
$ns duplex-link $n4 $n6 4000Mb 800ms RED

$ns queue-limit $n3 $n4 $queue_limit
$ns queue-limit $n4 $n3 $queue_limit

# Define the orientation of links for better visualization in NAM
$ns duplex-link-op $n1 $n3 orient right-down
$ns duplex-link-op $n2 $n3 orient right-up
$ns duplex-link-op $n3 $n4 orient right
$ns duplex-link-op $n4 $n5 orient right-up
$ns duplex-link-op $n4 $n6 orient right-down

# Configure the first TCP source (using the Linux Agent)
set source1 [new Agent/TCP/Linux]
$ns at 0.0 "$source1 select_ca yeah"
$source1 set class_ 2
$source1 set ttl_ 64
$source1 set windowSize_ 8
$source1 set window_ [expr {500 + ($seed % 500)}]
$source1 set packet_size_ 1000
$source1 set rto_ [expr {100 + ($seed % 200)}]

$ns attach-agent $n1 $source1
set sink1 [new Agent/TCPSink/Sack1]
$ns attach-agent $n5 $sink1
$ns connect $source1 $sink1
$source1 set fid_ 1

# Configure the second TCP source (also using the Linux Agent)
set source2 [new Agent/TCP/Linux]
$ns at 0.0 "$source2 select_ca yeah"
$source2 set class_ 1
$source2 set ttl_ 64
$source2 set windowSize_ 8
$source2 set window_ [expr {500 + (($seed + 100) % 500)}]
$source2 set packet_size_ 1000
$source2 set rto_ [expr {100 + (($seed + 100) % 200)}]

# Attach the second TCP source to node n2
$ns attach-agent $n2 $source2
set sink2 [new Agent/TCPSink/Sack1]
$ns attach-agent $n6 $sink2
$ns connect $source2 $sink2
$source2 set fid_ 2

# Attach the trace file to both TCP sources to log their internal variables
$source1 attach $tracefile1
$source1 tracevar cwnd_
$source1 tracevar ssthresh_
$source1 tracevar ack_
$source1 tracevar maxseq_
$source1 tracevar rtt_

$source2 attach $tracefile1
$source2 tracevar cwnd_ 
$source2 tracevar ssthresh_
$source2 tracevar ack_
$source2 tracevar maxseq_
$source2 tracevar rtt_

# Create FTP applications over the TCP connections
set myftp1 [new Application/FTP]
$myftp1 attach-agent $source1

set myftp2 [new Application/FTP]
$myftp2 attach-agent $source2

# Randomize the start times of the FTP applications to avoid perfect synchronization
set start1 [expr {0.1 + ($seed % 10) / 10.0}]
# Start time for the second flow: offset to ensure it's different
set start2 [expr {0.1 + (($seed + 50) % 10) / 10.0}]

$ns at $start1 "$myftp1 start"
$ns at $start2 "$myftp2 start"

$ns at 100.0 "finish"

$ns run
