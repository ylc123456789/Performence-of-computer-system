# Simulation Topology
#              n1                  n5
#               \                  /
#   4000Mb,500ms \   1000Mb,50ms  / 4000Mb,500ms
#              n3 --------------- n4
#   4000Mb,800ms /                \ 4000Mb,800ms
#               /                  \
#             n2                   n6 

set ns [new Simulator]

# --- 1. 接收随机种子（从环境变量SEED获取，默认用当前时间） ---
set seed [expr {[info exists ::env(SEED)] ? $::env(SEED) : [clock seconds]}]
puts "当前随机种子: $seed"  # 打印种子，验证是否生效

# --- 2. 读取带宽（保持原有逻辑，可被repeat_runs.sh覆盖） ---
set bw [expr {[info exists ::env(BW)] ? $::env(BW) : "500Mb"}]

$ns color 1 Blue
$ns color 2 Red

set namfile [open cubic.nam w]
$ns namtrace-all $namfile
set tracefile1 [open cubicTrace.tr w]
$ns trace-all $tracefile1

proc finish {} {
    global ns namfile
    $ns flush-trace
    close $namfile
    exit 0
}

set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

# --- 3. 拓扑参数随机化（基于seed） ---
# 瓶颈链路延迟随机化（50ms ± 20ms）
set bottleneck_delay [expr {50 + ($seed % 41) - 20}]  ;# 30~70ms随机
# 队列长度随机化（10 ± 5）
set queue_limit [expr {10 + ($seed % 11) - 5}]        ;# 5~15随机

$ns duplex-link $n1 $n3 4000Mb 500ms RED
$ns duplex-link $n2 $n3 4000Mb 800ms RED 
$ns duplex-link $n3 $n4 $bw ${bottleneck_delay}ms RED  ;# 应用随机延迟
$ns duplex-link $n4 $n5 4000Mb 500ms RED
$ns duplex-link $n4 $n6 4000Mb 800ms RED

$ns queue-limit $n3 $n4 $queue_limit  ;# 应用随机队列长度
$ns queue-limit $n4 $n3 $queue_limit

$ns duplex-link-op $n1 $n3 orient right-down
$ns duplex-link-op $n2 $n3 orient right-up
$ns duplex-link-op $n3 $n4 orient right
$ns duplex-link-op $n4 $n5 orient right-up
$ns duplex-link-op $n4 $n6 orient right-down

# --- 4. TCP参数随机化（source1） ---
set source1 [new Agent/TCP/Linux]
$ns at 0 "$source1 select_ca cubic"
$source1 set class_ 2
$source1 set ttl_ 64
$source1 set window_ [expr {500 + ($seed % 500)}]  ;# 初始窗口500~999随机
$source1 set packet_size_ 1000
$source1 set seed_ $seed  ;# 将种子传递给TCP Agent，触发内部随机逻辑
$source1 set rto_ [expr {100 + ($seed % 200)}]     ;# 重传超时100~299ms随机

$ns attach-agent $n1 $source1
set sink1 [new Agent/TCPSink/Sack1]
$ns attach-agent $n5 $sink1
$ns connect $source1 $sink1
$source1 set fid_ 1

# --- 5. TCP参数随机化（source2，种子偏移避免与source1完全同步） ---
set source2 [new Agent/TCP/Linux]
$ns at 0.0 "$source2 select_ca cubic"
$source2 set class_ 1
$source2 set ttl_ 64
$source2 set window_ [expr {500 + (($seed + 100) % 500)}]  ;# 偏移种子，窗口不同步
$source2 set packet_size_ 1000
$source2 set seed_ [expr {$seed + 100}]  ;# 种子偏移，行为更随机
$source2 set rto_ [expr {100 + (($seed + 100) % 200)}]     ;# 重传超时偏移

$ns attach-agent $n2 $source2
set sink2 [new Agent/TCPSink/Sack1]
$ns attach-agent $n6 $sink2
$ns connect $source2 $sink2
$source2 set fid_ 2

# --- 6. 流量启动时间随机化（避免同时启动导致行为一致） ---
set start1 [expr {0.1 + ($seed % 10) / 10.0}]    ;# 0.1~1.0秒随机
set start2 [expr {0.1 + (($seed + 50) % 10) / 10.0}]  ;# 偏移启动时间

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


set myftp1 [new Application/FTP]
$myftp1 attach-agent $source1


set myftp2 [new Application/FTP]
$myftp2 attach-agent $source2


$ns at $start1 "$myftp1 start"  ;# 应用随机启动时间
$ns at $start2 "$myftp2 start"  ;# 应用随机启动时间

$ns at 100.0 "finish"

$ns run
