set p1Counter 0
set p2Counter 0
set v1Counter 0
set v2Counter 0
set v3Counter 0
set p1 0
set p2 0
set v1 0
set v2 0
set v3 0
loop
wait
read var
rdata $var sender val
function y adapter ada,100,10
if($y==10.0)
	data p s $var
	send $p 46
end
if($y==20.0)
	if($sender==s34)
		plus p1Counter $p1Counter $val
		plus p1 $p1 1
	end
if($sender==s33)
	minus p1Counter $p1Counter $val
	plus p1 $p1 1
end
if ($p1>=2)
	if($p2Counter>=200)
		send N 35
	else
		send A 35
	end
	set p1 0
end

if($sender==s42)
	plus p2 $p2 1
	plus p2Counter $p2Counter $val
end
if($sender==s41)
	set p2ExBool 1
	plus p2 $p2 1
end
if ($p2>=2)
print here
	if($p2Counter>=300)
		send N 43
	else
		send A 43
	end
	set p2 0
end
if($sender==s1)
	plus v1 $v1 1
	plus v1Counter $v1Counter $val
end

if($sender==s2)
	plus v1 $v1 1
	minus v1Counter $v1Counter $val
end
if ($v1>=2)
print here
	if($v1Counter>=500)
		send N 7
	else
		send A 7
	end
	set v1 0
end

if($sender==s18)
	plus v2 $v2 1
	plus v2Counter $v2Counter $val
	end
if($sender==s20)
	plus v2 $v2 1
	minus v2Counter $v2Counter $val
end
if ($v2 >=2)
	if($v2Counter>=300)
		send N 17
	else
		send A 17
	end
	set v2 0
end
if($sender==s24)
	plus v3 $v3 1
	plus v3Counter $v3Counter $val
end
if($sender==s25)
	plus v3 $v3 1
	minus v3Counter $v3Counter $val
end
if ($v3 >=2)
	if($v3Counter>=300)
		send N 26
	else
		send A 26
	end
	set v3 0
end
data p s $var
send $p 46
end
if($y==30.0)
	data p s $var
	send $p 46
end

